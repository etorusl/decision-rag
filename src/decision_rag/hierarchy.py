"""Offline NQ hierarchy construction; embeddings are never searched at query time."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .dense_search import resolve_search_device, stable_torch_top_k
from .schema import Corpus, write_jsonl

SUMMARY_INSTRUCTION = (
    "Write one sentence (at most 30 words) describing what the pages in this group are about. "
    "Name the main kinds of subjects it covers, including the less typical ones, so that a reader "
    "can tell whether a given Wikipedia page would belong to this group."
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_sentence(text: str) -> str:
    """Deterministic lightweight sentence rule; not an unpublished author tokenizer."""
    normalized = " ".join(text.split())
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized, maxsplit=1)[0]


def page_features(corpus: Corpus) -> tuple[list, list[str]]:
    """Page title + first passage's first sentence, in stable document order."""
    pages = corpus.children(corpus.root.id)
    if not pages or any(page.kind != "page" for page in pages):
        raise ValueError("Cluster builder expects an unclustered root → page → passage corpus")
    texts = []
    for page in pages:
        passages = corpus.children(page.id)
        if not passages or any(node.kind != "passage" for node in passages):
            raise ValueError("NQ pages must contain passage children directly")
        title = page.title or page.text
        texts.append(f"query: {title}\n{first_sentence(passages[0].text)}")
    return pages, texts


def _release_gpu(device: str | None = None) -> None:
    gc.collect()
    if device is not None and not str(device).startswith('cuda'):
        return
    import torch
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def encode_pages(texts: list[str], model_name: str, device: str, batch_size: int,
                 revision: str | None = None) -> tuple[np.ndarray, dict]:
    """E5 mean pooling with mask, 512-token limit, and normalized vectors."""
    import torch
    from torch.nn import functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    vectors, truncated = [], 0
    try:
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset:offset + batch_size]
                truncated += sum(len(tokenizer.encode(text)) > 512 for text in batch)
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512,
                                   return_tensors="pt").to(device)
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                vectors.append(functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
                del inputs, hidden, mask, pooled
        metadata = {"model": model_name, "requested_revision": revision,
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                    "pooling": "attention-mask mean, L2 normalization", "prefix": "query: ",
                    "max_tokens": 512, "truncated_pages": truncated}
        return np.concatenate(vectors).astype(np.float32), metadata
    finally:
        del model, tokenizer
        _release_gpu(device)


def _numeric_device(device):
    try:
        return resolve_search_device(device)
    except RuntimeError as exc:
        raise RuntimeError('GPU hierarchy preprocessing needs a working CUDA environment; '
                           'use --device cpu only for an explicitly separate CPU experiment') from exc


def fit_clusters(embeddings: np.ndarray, clusters: int, seed: int,
                 n_init: int = 10, max_iter: int = 300, *, device: str = 'cuda') -> tuple[np.ndarray, np.ndarray, dict]:
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    if not 1 <= clusters <= len(embeddings):
        raise ValueError("clusters must be between 1 and the number of pages")
    if min(n_init, max_iter) < 1:
        raise ValueError('n_init and max_iter must be positive')
    device = _numeric_device(device)
    common = {'seed': seed, 'n_init': n_init, 'max_iter': max_iter, 'device': device}
    if device == 'cpu':
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=clusters, random_state=seed, n_init=n_init,
                       max_iter=max_iter, algorithm='lloyd')
        labels = model.fit_predict(embeddings)
        centroids = model.cluster_centers_
        metadata = {**common, 'backend': 'scikit-learn', 'algorithm': 'KMeans/lloyd',
                    'init': 'k-means++', 'precision': str(embeddings.dtype),
                    'iterations': int(model.n_iter_), 'inertia': float(model.inertia_)}
    else:
        try:
            import cupy
            from cuml.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError("GPU clustering requires cuML: pip install -e '.[gpu-clustering]' "
                               '--extra-index-url https://pypi.nvidia.com; no CPU fallback was performed') from exc
        model = None
        with cupy.cuda.Device(int(device.split(':')[1])) as gpu:
            try:
                model = KMeans(n_clusters=clusters, random_state=seed, n_init=n_init,
                               max_iter=max_iter, init='scalable-k-means++', output_type='numpy')
                model.fit(np.asarray(embeddings, dtype=np.float32))
                labels = np.asarray(model.labels_, dtype=np.int64)
                centroids = np.asarray(model.cluster_centers_, dtype=np.float32)
                iterations = getattr(model, 'n_iter_', None)
                metadata = {**common, 'backend': 'cuML', 'algorithm': 'cuML/KMeans',
                            'init': 'scalable-k-means++', 'precision': 'float32',
                            'iterations': int(iterations) if iterations is not None else None,
                            'inertia': float(model.inertia_)}
                gpu.synchronize()
            finally:
                # Release the clustering estimator and allocator scratch before the summary LM.
                del model
                gc.collect()
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
    validate_labels(labels, len(embeddings), clusters)
    return labels, centroids, metadata


def validate_labels(labels: np.ndarray, pages: int, clusters: int) -> None:
    if labels.shape != (pages,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Cluster assignment must have one integer label per page")
    if set(labels.tolist()) != set(range(clusters)):
        raise ValueError("Every requested cluster must be nonempty, with labels 0 through clusters-1")


def recompute_centroids(embeddings: np.ndarray, labels: np.ndarray, clusters: int,
                        *, device: str = 'cuda') -> np.ndarray:
    """Rebuild centers for saved assignments on the selected numeric device."""
    validate_labels(labels, len(embeddings), clusters)
    device = _numeric_device(device)
    if device == 'cpu':
        return np.stack([embeddings[labels == i].mean(axis=0) for i in range(clusters)])
    import torch

    vectors = assignments = centers = counts = None
    try:
        with torch.inference_mode():
            vectors = torch.as_tensor(embeddings, device=device, dtype=torch.float32)
            assignments = torch.as_tensor(labels, device=device, dtype=torch.long)
            centers = torch.zeros((clusters, embeddings.shape[1]), device=device, dtype=torch.float32)
            centers.index_add_(0, assignments, vectors)
            counts = torch.bincount(assignments, minlength=clusters)
            centers /= counts[:, None]
            result = centers.cpu().numpy()
            torch.cuda.synchronize(device)
            return result
    finally:
        del vectors, assignments, centers, counts
        _release_gpu(device)


def typical_titles(pages, embeddings: np.ndarray, labels: np.ndarray,
                   centroids: np.ndarray, max_titles: int = 300, *, device: str = 'cuda') -> dict[str, list[str]]:
    if not 1 <= max_titles <= 300:
        raise ValueError("max_titles must be between 1 and 300")
    validate_labels(labels, len(embeddings), len(centroids))
    device = _numeric_device(device)
    # Group integer labels once; sorting page-ID strings defines deterministic tie order.
    groups = [[] for _ in range(len(centroids))]
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    titles = {}
    if device == 'cpu':
        for cluster, group in enumerate(groups):
            members = np.asarray(sorted(group, key=lambda i: pages[i].id))
            distances = np.sum((embeddings[members] - centroids[cluster]) ** 2, axis=1)
            order = np.argsort(distances, kind='stable')[:max_titles]
            titles[str(cluster)] = [pages[members[i]].title or pages[members[i]].text for i in order]
        return titles
    import torch

    vectors = centers = members = distances = order = None
    try:
        with torch.inference_mode():
            vectors = torch.as_tensor(embeddings, device=device, dtype=torch.float32)
            centers = torch.as_tensor(centroids, device=device, dtype=torch.float32)
            for cluster, group in enumerate(groups):
                by_id = sorted(group, key=lambda i: pages[i].id)
                members = torch.tensor(by_id, device=device, dtype=torch.long)
                distances = (vectors[members] - centers[cluster]).square().sum(dim=1)
                order = stable_torch_top_k(-distances, max_titles).cpu().tolist()
                titles[str(cluster)] = [pages[by_id[i]].title or pages[by_id[i]].text for i in order]
            torch.cuda.synchronize(device)
    finally:
        del vectors, centers, members, distances, order
        _release_gpu(device)
    return titles


def _compact_summary(raw: str) -> tuple[str, bool]:
    # Qwen thinking is disabled, but remove an unexpected completed block defensively.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        raise ValueError("Summary model returned an unfinished thinking block")
    cleaned = first_sentence(cleaned).strip(' \"')
    words = cleaned.split()
    if not words:
        raise ValueError("Summary model returned an empty description")
    clipped = len(words) > 30
    if clipped:
        cleaned = " ".join(words[:30]).rstrip(".,;:") + "."
    return cleaned, clipped


def summarize_clusters(titles: dict[str, list[str]], model_name: str, device: str,
                       revision: str | None = None, max_input_tokens: int = 16384) -> tuple[dict, dict]:
    """Generate descriptions offline, after the embedding model has been released."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                torch_dtype=dtype).to(device).eval()
    summaries, details = {}, {}
    context_limit = min(max_input_tokens, getattr(model.config, "max_position_embeddings", max_input_tokens) - 96)
    try:
        with torch.inference_mode():
            for cluster, candidates in titles.items():
                kept = list(candidates)
                while kept:
                    prompt = SUMMARY_INSTRUCTION + "\n\nPage titles:\n" + "\n".join(kept)
                    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                            tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False)
                    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
                    if inputs.input_ids.shape[1] <= context_limit:
                        break
                    kept.pop()
                if not kept:
                    raise ValueError(f"Even one title in cluster {cluster} exceeds the summary input budget")
                tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                summary, clipped = _compact_summary(raw)
                summaries[cluster] = summary
                details[cluster] = {"titles_used": len(kept), "input_tokens": int(inputs.input_ids.shape[1]),
                                    "raw_output": raw, "clipped_to_30_words": clipped,
                                    "dropped_titles_for_context": len(candidates) - len(kept)}
                del inputs, tokens
        return summaries, {"model": model_name, "requested_revision": revision,
                           "resolved_revision": getattr(model.config, "_commit_hash", None),
                           "thinking": False, "do_sample": False, "max_new_tokens": 96,
                           "max_input_tokens": max_input_tokens, "clusters": details}
    finally:
        del model, tokenizer
        _release_gpu(device)


def _validate_summaries(summaries: dict, clusters: int) -> None:
    if not isinstance(summaries, dict) or set(summaries) != {str(i) for i in range(clusters)}:
        raise ValueError("Summaries must include exactly one description per cluster label")
    for summary in summaries.values():
        if not isinstance(summary, str) or not 1 <= len(summary.split()) <= 30:
            raise ValueError("Each cluster summary must be a nonempty string of at most 30 words")


def cluster_corpus(input_path: str | Path, output_path: str | Path,
                   model_name: str = "intfloat/e5-base-v2", clusters: int = 1000,
                   summary_model: str = "Qwen/Qwen3-8B", device: str = "cuda",
                   batch_size: int = 64, seed: int = 42, *, summaries_path: str | Path | None = None,
                   assignments_path: str | Path | None = None, embeddings_path: str | Path | None = None,
                   model_revision: str | None = None, summary_revision: str | None = None,
                   max_titles: int = 300, n_init: int = 10, max_iter: int = 300,
                   max_summary_tokens: int = 16384) -> dict:
    """Write a clustered corpus and reusable, hash-bound preprocessing artifacts.

    Model calls are local only. Reuse assignments and summaries together to avoid
    regeneration and guarantee the same tree; passage and page IDs never change.
    """
    started = time.perf_counter()
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output corpus paths must differ")
    if min(clusters, batch_size, n_init, max_iter, max_summary_tokens) <= 0 or not 1 <= max_titles <= 300:
        raise ValueError("Invalid clustering or inference budget")
    corpus = Corpus.load(input_path)
    pages, texts = page_features(corpus)
    if clusters > len(pages):
        raise ValueError("clusters exceeds the number of pages")
    source_hash = file_sha256(input_path)
    page_ids = [page.id for page in pages]
    identity = {"source_sha256": source_hash, "page_ids_sha256": _hash_json(page_ids)}
    artifact_dir = output_path.parent / f"{output_path.stem}.hierarchy"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timing = {"load_seconds": time.perf_counter() - started}
    embed_info, cluster_info, summary_info = {}, {}, {}
    embeddings, centroids = None, None

    if assignments_path:
        assignment = json.loads(Path(assignments_path).read_text())
        if any(assignment.get(key) != value for key, value in identity.items()) or assignment.get("page_ids") != page_ids:
            raise ValueError("Supplied assignments do not match the input corpus and page order")
        labels = np.asarray(assignment["labels"])
        if assignment.get("cluster_count") != clusters:
            raise ValueError("Supplied assignment cluster count differs from requested clusters")
        validate_labels(labels, len(pages), clusters)
        cluster_info = {"reused_assignments": str(assignments_path), "sha256": file_sha256(assignments_path)}
    else:
        stage_started = time.perf_counter()
        if embeddings_path:
            embeddings = np.load(embeddings_path, allow_pickle=False)
            sidecar = json.loads(Path(str(embeddings_path) + ".json").read_text())
            if any(sidecar.get(key) != value for key, value in identity.items()):
                raise ValueError("Supplied embeddings do not match the input corpus and page order")
            embed_info = {"reused_embeddings": str(embeddings_path), "sha256": file_sha256(embeddings_path)}
        else:
            embeddings, embed_info = encode_pages(texts, model_name, device, batch_size, model_revision)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(pages) or not np.isfinite(embeddings).all():
            raise ValueError("Embedding count/values do not match the corpus")
        vector_file = artifact_dir / "embeddings.npy"
        np.save(vector_file, embeddings, allow_pickle=False)
        _write_json(Path(str(vector_file) + ".json"), {**identity, **embed_info})
        embed_info["artifact_sha256"] = file_sha256(vector_file)
        timing["embedding_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        labels, centroids, cluster_info = fit_clusters(embeddings, clusters, seed, n_init, max_iter, device=device)
        timing["kmeans_seconds"] = time.perf_counter() - stage_started

    assignment_hash = _hash_json(labels.tolist())
    assignment = {"schema_version": 1, **identity, "cluster_count": clusters, "page_ids": page_ids,
                  "labels": labels.tolist(), "assignment_sha256": assignment_hash}
    _write_json(artifact_dir / "assignments.json", assignment)
    stage_started = time.perf_counter()
    if summaries_path:
        supplied = json.loads(Path(summaries_path).read_text())
        if "summaries" not in supplied:
            raise ValueError("Supplied summaries must use the hash-bound summaries.json artifact format")
        if any(supplied.get(key) != value for key, value in identity.items()) or supplied.get("assignment_sha256") != assignment_hash:
            raise ValueError("Supplied summaries do not match corpus/cluster assignments")
        summaries = supplied["summaries"]
        summary_info = {"reused_summaries": str(summaries_path), "sha256": file_sha256(summaries_path)}
    else:
        if embeddings is None:
            # Assignments alone cannot recover typicality; recompute vectors without changing labels.
            embeddings, embed_info = encode_pages(texts, model_name, device, batch_size, model_revision)
        if centroids is None:
            centroids = recompute_centroids(embeddings, labels, clusters, device=device)
            cluster_info['centroid_reduction'] = 'numpy mean' if device == 'cpu' else 'torch index_add/bincount FP32'
        titles = typical_titles(pages, embeddings, labels, centroids, max_titles, device=device)
        _write_json(artifact_dir / "summary_titles.json", titles)
        summaries, summary_info = summarize_clusters(titles, summary_model, device, summary_revision,
                                                     max_summary_tokens)
        summary_info['title_selection'] = {'device': _numeric_device(device),
                                           'distance': 'squared Euclidean', 'tie_break': 'page ID ascending',
                                           'precision': 'float32' if device != 'cpu' else str(embeddings.dtype)}
    _validate_summaries(summaries, clusters)
    timing["summary_seconds"] = time.perf_counter() - stage_started
    _write_json(artifact_dir / "summaries.json", {"schema_version": 1, **identity,
                                                "assignment_sha256": assignment_hash, "summaries": summaries})
    cluster_ids = [f"decisionrag:cluster:{i}" for i in range(clusters)]
    if set(cluster_ids) & set(corpus.nodes):
        raise ValueError("Generated cluster IDs collide with existing corpus IDs")
    from .schema import Node

    def nodes():
        yield asdict(replace(corpus.root, metadata={**corpus.root.metadata, "hierarchy": "e5-kmeans-qwen",
                                                    "hierarchy_assignment_sha256": assignment_hash}))
        for i, cluster_id in enumerate(cluster_ids):
            yield asdict(Node(cluster_id, summaries[str(i)], "group", corpus.root.id,
                              title=f"Topic {i + 1}", order=i,
                              metadata={"cluster_label": i, "page_count": int(np.sum(labels == i))}))
        for page, cluster in zip(pages, labels):
            yield asdict(replace(page, parent_id=cluster_ids[int(cluster)]))
            for passage in corpus.children(page.id):
                yield asdict(passage)
    write_jsonl(output_path, nodes())
    timing["total_seconds"] = time.perf_counter() - started
    versions = {}
    for package in ("numpy", "torch", "transformers", "scikit-learn", "cuml-cu12", "cupy-cuda12x"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sizes = np.bincount(labels, minlength=clusters)
    metadata = {"schema_version": 1, **identity, "output_sha256": file_sha256(output_path),
                "output_path": str(output_path), "artifact_dir": str(artifact_dir),
                "pages": len(pages), "passages": len(corpus.passages), "clusters": clusters,
                "cluster_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
                "assignment_sha256": assignment_hash, "embedding": embed_info, "clustering": cluster_info,
                "summarizer": summary_info, "summary_instruction": SUMMARY_INSTRUCTION,
                "max_titles": max_titles, "device": device, "batch_size": batch_size,
                "timings": timing, "versions": versions,
                "reproduction_note": "Independent seeded hierarchy; cluster assignments and 500-query IDs are not specified in the supplied paper."}
    _write_json(artifact_dir / "metadata.json", metadata)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--summary-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--clusters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--summaries-path")
    parser.add_argument("--assignments-path")
    parser.add_argument("--embeddings-path")
    parser.add_argument("--model-revision")
    parser.add_argument("--summary-revision")
    parser.add_argument("--max-titles", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-summary-tokens", type=int, default=16384)
    result = cluster_corpus(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("output_path", "artifact_dir", "pages", "clusters", "timings")}, indent=2))


if __name__ == "__main__":
    main()
