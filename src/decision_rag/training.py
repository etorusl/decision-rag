"""Train a local decision head on explicit labeled JSONL requests.

This implements the Appendix G scoring architecture, not the authors' unreleased
multi-dataset general-training checkpoint or data-generation recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .backends.decision_head import (
    DecisionModel,
    check_layout,
    check_model,
    encode_record,
    resolve_attention,
    resolve_dtype,
)


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-8B-Base"
    mode: str = "lora"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention: str = "auto"
    epochs: int = 1
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    head_only_steps: int = 0
    head_only_learning_rate: float = 1e-3
    warmup_steps: int = 50
    gradient_accumulation: int = 4
    max_steps: int | None = None
    max_context_tokens: int = 32768
    max_packed_tokens: int = 40960
    projection_dim: int = 1024
    lora_rank: int = 64
    lora_alpha: int = 128
    option_links: bool = False
    seed: int = 42
    save_every: int = 500


class JsonlRecords:
    """Byte offsets keep large contexts off the Python heap between training steps."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No training records in {path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        questions, options, labels = row["questions"], row["options"], row["labels"]
        if not isinstance(row["context"], str) or not questions:
            raise ValueError(f"Record {index}: non-empty questions and string context required")
        if not (len(questions) == len(options) == len(labels)):
            raise ValueError(f"Record {index}: question, option and label lengths differ")
        for choices, label in zip(options, labels, strict=True):
            if len(choices) < 2 or not isinstance(label, int) or not 0 <= label < len(choices):
                raise ValueError(f"Record {index}: invalid answer options/label")
        return row


def _record_layout(tokenizer, row, config):
    layout = encode_record(
        tokenizer, row["context"], row["questions"], row["options"], option_links=config.option_links
    )
    check_layout(layout, config.max_context_tokens, config.max_packed_tokens)
    return layout


def _loss(model, layout, labels, attention):
    logits = model(layout, attention)
    if len(logits) != len(labels):
        raise ValueError("Logit/label counts differ")
    if len({row.numel() for row in logits}) == 1:
        loss = F.cross_entropy(torch.stack(logits), torch.tensor(labels, device=logits[0].device))
    else:
        loss = torch.stack(
            [
                F.cross_entropy(scores.unsqueeze(0), torch.tensor([label], device=scores.device))
                for scores, label in zip(logits, labels, strict=True)
            ]
        ).mean()
    return loss, logits


def _save(model, tokenizer, output: Path, config, steps: int, extra: dict):
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output / "tokenizer")
    if config.mode == "lora":
        model.backbone.save_pretrained(output / "adapter", safe_serialization=True)
    elif config.mode == "full":
        model.backbone.save_pretrained(output / "backbone", safe_serialization=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()},
        str(output / "head.safetensors"),
    )
    manifest = {
        "format_version": 1,
        "base_model": config.model_name,
        "mode": config.mode,
        "projection_dim": config.projection_dim,
        "option_links": config.option_links,
        "trained_steps": steps,
        "training_config": asdict(config),
        **extra,
    }
    (output / "decision_config.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train_records(
    train_path: str | Path,
    output_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
    dev_path: str | Path | None = None,
    **overrides,
) -> dict:
    """Train a checkpoint; keyword overrides are TrainingConfig fields.

    JSONL: {context, questions:[str], options:[[str]], labels:[zero-based int]}.
    Full records that exceed the configured budgets are dropped, never truncated.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if config is not None and overrides:
        raise ValueError("Pass a TrainingConfig or keyword settings, not both")
    config = config or TrainingConfig(**overrides)
    if Path(config.model_name).exists():
        config.model_name = str(Path(config.model_name).resolve())
    if config.mode not in {"lora", "head", "full"}:
        raise ValueError("mode must be lora, head or full")
    if config.epochs < 1 or config.gradient_accumulation < 1 or config.head_only_steps < 0:
        raise ValueError("Invalid epoch/gradient accumulation/head warm-up setting")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; use a new checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    attention = resolve_attention(config.device, config.attention)
    model_config = AutoConfig.from_pretrained(config.model_name)
    config.max_context_tokens = min(config.max_context_tokens, model_config.max_position_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    records = JsonlRecords(train_path)
    eligible, dropped = [], 0
    for i in range(len(records)):
        row = records[i]
        try:
            layout = _record_layout(tokenizer, row, config)
            if attention != "flex_attention" and layout.length > 4096:
                raise ValueError("CPU dense mask length limit")
        except ValueError as exc:
            if not any(word in str(exc) for word in ("exceeds", "limit", "Packed")):
                raise
            dropped += 1
            continue
        eligible.append(i)
    if not eligible:
        raise ValueError(
            "All training records exceed token budgets; increase limits or prepare smaller requests"
        )
    print(
        json.dumps({"event": "preflight", "eligible_records": len(eligible), "dropped_overlong": dropped}),
        flush=True,
    )
    backbone = AutoModel.from_pretrained(
        config.model_name,
        torch_dtype=resolve_dtype(config.dtype),
        attn_implementation=attention,
    )
    check_model(backbone)
    if config.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    elif config.mode == "head":
        backbone.requires_grad_(False)
    if config.mode != "head":
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    model = DecisionModel(backbone, config.projection_dim).to(config.device)
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    all_trainable = [*backbone_parameters, *model.head.parameters()]
    summary = {
        "source_train": str(Path(train_path).resolve()),
        "source_dev": str(Path(dev_path).resolve()) if dev_path else None,
        "eligible_records": len(eligible),
        "dropped_overlong_records": dropped,
        "trainable_parameters": sum(p.numel() for p in all_trainable),
        "reproduction": "Appendix G architecture; independent token format and training recipe",
        "base_revision": getattr(backbone.config, "_commit_hash", None),
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ["torch", "transformers", "safetensors"]
        },
    }

    def sha256(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    summary["train_sha256"] = sha256(train_path)
    summary["dev_sha256"] = sha256(dev_path) if dev_path else None
    (output / "training_config.json").write_text(json.dumps({**asdict(config), **summary}, indent=2) + "\n")
    steps, processed = 0, 0
    log_path = output / "train_log.jsonl"

    def log(row):
        row = {"step": steps, **row}
        with log_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    # Optional stable head initialization while retaining the same packed examples.
    if config.head_only_steps:
        for p in backbone_parameters:
            p.requires_grad_(False)
        model.backbone.eval()
        model.head.train()
        head_optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=config.head_only_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
        )
        for step in range(config.head_only_steps):
            row = records[eligible[step % len(eligible)]]
            layout = _record_layout(tokenizer, row, config)
            head_optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(model, layout, row["labels"], attention)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            head_optimizer.step()
            steps += 1
            log({"phase": "head_warmup", "loss": float(loss.detach())})
        for p in backbone_parameters:
            p.requires_grad_(True)
        del head_optimizer

    groups = [{"params": list(model.head.parameters()), "lr": config.learning_rate}]
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.learning_rate if config.mode == "lora" else config.backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = math.ceil(len(eligible) / config.gradient_accumulation) * config.epochs
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)

    def multiplier(step):
        warmup = min(config.warmup_steps, max(total_steps - 1, 0))
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    updates = 0
    for epoch in range(config.epochs):
        order = list(eligible)
        random.Random(config.seed + epoch).shuffle(order)
        model.train()
        if config.mode == "head":
            model.backbone.eval()
        for start in range(0, len(order), config.gradient_accumulation):
            batch = order[start : start + config.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for index in batch:
                row = records[index]
                layout = _record_layout(tokenizer, row, config)
                loss, _ = _loss(model, layout, row["labels"], attention)
                (loss / len(batch)).backward()
                batch_loss += float(loss.detach()) / len(batch)
                processed += 1
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
            updates += 1
            steps += 1
            log(
                {
                    "phase": "train",
                    "epoch": epoch + 1,
                    "loss": batch_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "processed_records": processed,
                }
            )
            if config.save_every > 0 and updates % config.save_every == 0:
                _save(model, tokenizer, output / f"step-{steps}", config, steps, summary)
            if updates >= total_steps:
                break
        if updates >= total_steps:
            break
    if dev_path:
        dev = JsonlRecords(dev_path)
        losses, correct, count, dev_dropped = [], 0, 0, 0
        model.eval()
        with torch.inference_mode():
            for i in range(len(dev)):
                row = dev[i]
                try:
                    layout = _record_layout(tokenizer, row, config)
                    if attention != "flex_attention" and layout.length > 4096:
                        raise ValueError("Dense mask limit")
                except ValueError as exc:
                    if not any(word in str(exc) for word in ("exceeds", "limit", "Packed")):
                        raise
                    dev_dropped += 1
                    continue
                loss, logits = _loss(model, layout, row["labels"], attention)
                losses.append(float(loss))
                correct += sum(int(scores.argmax()) == label for scores, label in zip(logits, row["labels"]))
                count += len(row["labels"])
        summary["validation"] = {
            "mean_record_loss": sum(losses) / len(losses) if losses else None,
            "accuracy": correct / count if count else None,
            "questions": count,
            "dropped_overlong_records": dev_dropped,
        }
        log({"phase": "validation", **summary["validation"]})
    summary.update({"trained_steps": steps, "optimizer_updates": updates, "processed_records": processed})
    _save(model, tokenizer, output, config, steps, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=TrainingConfig.model_name)
    parser.add_argument("--mode", choices=["lora", "head", "full"], default="lora")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-only-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-packed-tokens", type=int, default=40960)
    parser.add_argument("--projection-dim", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--option-links", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500)
    args = vars(parser.parse_args(argv))
    train, dev, output = args.pop("train"), args.pop("dev"), args.pop("output")
    args["model_name"] = args.pop("model")
    print(json.dumps(train_records(train, output, dev_path=dev, **args), indent=2))


if __name__ == "__main__":
    main()
