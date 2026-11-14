"""Check the local CUDA stack and packed branch attention without downloading weights.

Run after installing the project's local extra: python scripts/h100_preflight.py
The random tiny Qwen models check numerical/gradient correctness, not retrieval quality.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "triton", "peft", "accelerate", "safetensors", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_attention(device: str, dtype_name: str, option_links: bool,
                    gradient_checkpointing: bool) -> dict:
    import torch
    from torch.nn import functional
    from transformers import Qwen3Config, Qwen3Model

    from decision_rag.backends.decision_head import DecisionModel, pack_branches

    torch.manual_seed(73)
    dtype = getattr(torch, dtype_name)
    kwargs = {
        "vocab_size": 128, "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "max_position_embeddings": 2048, "attention_dropout": 0.0,
        "pad_token_id": 0, "eos_token_id": 1,
    }
    dense_config = Qwen3Config(**kwargs, _attn_implementation="sdpa")
    flex_config = Qwen3Config(**kwargs, _attn_implementation="flex_attention")
    dense = DecisionModel(Qwen3Model(dense_config).to(dtype=dtype), projection_dim=32).to(device)
    flex = DecisionModel(Qwen3Model(flex_config).to(dtype=dtype), projection_dim=32).to(device)
    flex.load_state_dict(dense.state_dict())
    if gradient_checkpointing:
        for model in (dense, flex):
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.backbone.enable_input_require_grads()
    # Exercise more than one 128-token mask block and a nonmultiple final block.
    prefix = [2 + i % 100 for i in range(257)]
    layout = pack_branches(prefix, [[7, 8, 9], [10, 11, 12, 13, 14], [15, 16]],
                           [[[20, 21], [22, 23, 24]], [[25], [26, 27]], [[28, 29, 30], [31]]],
                           option_links=option_links)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    dense_logits = torch.stack(dense(layout, "sdpa"))
    flex_logits = torch.stack(flex(layout, "flex_attention"))
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    atol, rtol = (5e-2, 3e-2) if dtype_name == "bfloat16" else (3e-4, 3e-3)
    torch.testing.assert_close(flex_logits, dense_logits, atol=atol, rtol=rtol)
    labels = torch.tensor([0, 1, 0], device=device)
    functional.cross_entropy(dense_logits, labels).backward()
    functional.cross_entropy(flex_logits, labels).backward()
    torch.cuda.synchronize(device)
    gradients = {}
    for name in ("head.question_projection.weight", "head.option_projection.weight",
                 "backbone.layers.0.self_attn.q_proj.weight", "backbone.embed_tokens.weight"):
        dense_grad = dense.get_parameter(name).grad
        flex_grad = flex.get_parameter(name).grad
        if dense_grad is None or flex_grad is None or not torch.isfinite(flex_grad).all():
            raise RuntimeError(f"Missing or nonfinite gradient: {name}")
        torch.testing.assert_close(flex_grad, dense_grad, atol=atol, rtol=rtol)
        gradients[name] = float((flex_grad.float() - dense_grad.float()).abs().max())
    result = {"status": "passed", "dtype": dtype_name, "option_links": option_links,
              "gradient_checkpointing": gradient_checkpointing, "packed_tokens": layout.length,
              "forward_seconds_including_compile": forward_seconds,
              "logit_max_absolute_difference": float((flex_logits - dense_logits).abs().max()),
              "gradient_max_absolute_differences": gradients,
              "absolute_tolerance": atol, "relative_tolerance": rtol}
    del dense, flex, dense_logits, flex_logits
    torch.cuda.empty_cache()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true", help="Print environment and skip GPU checks if unavailable")
    parser.add_argument("--json-output", type=Path, help="Save diagnostic JSON for experiment provenance")
    parser.add_argument("--skip-kernel-check", action="store_true", help="Only report environment; does not validate CUDA")
    args = parser.parse_args(argv)
    report = {"python": sys.version, "platform": platform.platform(), "versions": package_versions(),
              "requested_device": args.device, "model_downloads": False, "checks": []}
    result_code = 0
    try:
        import torch
        report.update({"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if not torch.cuda.is_available():
            report["status"] = "CUDA checks skipped: no CUDA device"
            result_code = 0 if args.allow_cpu else 2
        else:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            report.update({"gpu": properties.name, "compute_capability": list(torch.cuda.get_device_capability(device)),
                           "total_memory_gib": properties.total_memory / 1024**3,
                           "bf16_supported": torch.cuda.is_bf16_supported(),
                           "cudnn_version": torch.backends.cudnn.version()})
            if args.skip_kernel_check:
                report["status"] = "Environment only; kernels not checked"
            else:
                if not report["bf16_supported"]:
                    raise RuntimeError("This stack lacks the BF16 support required by the H100 runbook")
                from packaging.version import Version
                if Version(report["versions"]["torch"].split("+")[0]) < Version("2.7"):
                    raise RuntimeError("Packed FlexAttention requires PyTorch >=2.7")
                version = Version(report["versions"]["transformers"])
                if version < Version("4.57") or version >= Version("4.58"):
                    raise RuntimeError("Use the tested Transformers 4.57.x series for dictionary branch masks")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.cuda.reset_peak_memory_stats(device)
                for dtype, links, checkpointing in (("float32", False, False),
                                                    ("bfloat16", True, True)):
                    print(f"Checking {dtype}, option_links={links}, checkpointing={checkpointing}...", flush=True)
                    report["checks"].append(check_attention(args.device, dtype, links, checkpointing))
                report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                report["status"] = "Tiny CUDA FlexAttention forward/backward parity passed"
    except Exception as exc:  # noqa: BLE001 - diagnostic must serialize any CUDA/compiler failure
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
        result_code = 1
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
