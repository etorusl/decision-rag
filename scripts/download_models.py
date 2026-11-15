#!/usr/bin/env python3
"""Download benchmark weights once and record immutable revisions for all runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from decision_rag.observability import RunReporter, add_logging_args


def model_files(siblings):
    """Prefer one root weight format; omit duplicate ONNX/OpenVINO/training files."""
    names = sorted(item.rfilename for item in siblings)
    weights = [name for name in names if '/' not in name and name.endswith('.safetensors')]
    weight_format = 'safetensors'
    if not weights:
        weights = [name for name in names if '/' not in name and name.startswith('pytorch_model')
                   and name.endswith('.bin')]
        weight_format = 'pytorch'
    if not weights:
        raise ValueError('No standard Hugging Face model weights found in the repository root')
    metadata = [name for name in names if Path(name).suffix in {'.json', '.txt', '.model', '.tiktoken', '.jinja'}
                and not name.startswith(('onnx/', 'openvino/', 'coreml/'))]
    if 'config.json' not in metadata:
        raise ValueError('Model repository is missing config.json')
    # Weight index files from an unused format must not point at omitted shards.
    metadata = [name for name in metadata if not (
        weight_format == 'safetensors' and name.startswith('pytorch_model')
        or weight_format == 'pytorch' and name.startswith('model.safetensors'))]
    return sorted(set(metadata + weights)), weight_format


def download_models(models, manifest_path, *, cache_dir=None, refresh=False, max_workers=8,
                    progress='auto', log_interval=30.0):
    from huggingface_hub import HfApi, constants, snapshot_download

    if max_workers < 1:
        raise ValueError('max_workers must be positive')
    manifest_path = Path(manifest_path)
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('format_version') != 1 or not isinstance(previous.get('models'), dict):
            raise ValueError('Invalid model manifest; choose a new --manifest path')
    cache = Path(cache_dir or previous.get('cache_dir') or constants.HF_HUB_CACHE).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError('At least one model is required')
    api = HfApi()
    entries = {}
    with RunReporter(manifest_path.parent / 'model-download', progress=progress,
                     log_interval=log_interval, name='model-download') as reporter, \
            reporter.phase('Downloading models', total=len(models), unit='model') as phase:
        for model in models:
            pinned = previous.get('models', {}).get(model, {}).get('revision') if not refresh else None
            if pinned is not None and not re.fullmatch('[0-9a-fA-F]{40}', pinned):
                raise ValueError(f'Invalid saved revision for {model}')
            info = api.model_info(model, revision=pinned, files_metadata=True)
            revision = info.sha
            if not isinstance(revision, str) or not re.fullmatch('[0-9a-fA-F]{40}', revision):
                raise ValueError(f'Hugging Face did not resolve {model} to an immutable commit')
            if pinned and revision != pinned:
                raise ValueError(f'Hugging Face resolved a different revision for pinned model {model}')
            files, weight_format = model_files(info.siblings)
            reporter.event('download_started', f'Downloading {model}', model=model, revision=revision,
                           files=len(files), weight_format=weight_format, cache_dir=str(cache))
            snapshot = Path(snapshot_download(repo_id=model, revision=revision, cache_dir=str(cache),
                                             allow_patterns=files, max_workers=max_workers)).resolve()
            missing = [name for name in files if not (snapshot / name).is_file()]
            if missing:
                raise RuntimeError(f'Incomplete cached snapshot for {model}: {missing[:5]}')
            sizes = {item.rfilename: getattr(item, 'size', None) for item in info.siblings}
            entries[model] = {'revision': revision, 'snapshot_path': str(snapshot),
                              'weight_format': weight_format, 'files': files,
                              'size_bytes': sum(sizes[name] for name in files)
                              if all(sizes[name] is not None for name in files) else None}
            reporter.event('download_completed', f'Cached {model}', model=model, **entries[model])
            phase.update(model=model)
        manifest = {'format_version': 1, 'created_at': datetime.now(UTC).isoformat(),
                    'cache_dir': str(cache), 'models': entries}
        # An interrupted download must never replace a complete existing manifest.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=manifest_path.parent,
                                             prefix=manifest_path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            temporary.replace(manifest_path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        reporter.event('manifest_saved', f'Model manifest: {manifest_path}', models=len(entries))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='cache/model-manifest.json')
    parser.add_argument('--cache-dir', help='Hugging Face cache directory; defaults to HF_HUB_CACHE/HF_HOME')
    parser.add_argument('--refresh', action='store_true', help='Resolve new model revisions instead of reusing saved pins')
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--models', nargs='+', metavar='MODEL',
                        help='Download exactly these model IDs instead of the default benchmark suite')
    role_defaults = {
        'naive_model': os.getenv('NAIVE_MODEL', 'Qwen/Qwen3-8B'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B'),
        'reranker_model': os.getenv('RERANKER_MODEL', 'Qwen/Qwen3-Reranker-8B'),
        'bge_embedding_model': 'BAAI/bge-m3', 'bge_reranker_model': 'BAAI/bge-reranker-v2-m3',
    }
    for role, default in role_defaults.items():
        parser.add_argument('--' + role.replace('_', '-'), help=f'Default benchmark suite model (default: {default})')
    parser.add_argument('--skip-bge', action='store_true', help='Download only the original Qwen suite')
    add_logging_args(parser)
    args = parser.parse_args(argv)
    if args.models is not None:
        if args.skip_bge or any(getattr(args, role) is not None for role in role_defaults):
            parser.error('--models cannot be combined with --skip-bge or role-specific model options')
        models = args.models
    else:
        models = [getattr(args, role) if getattr(args, role) is not None else default
                  for role, default in role_defaults.items() if not (args.skip_bge and role.startswith('bge_'))]
    if any(not model.strip() for model in models):
        parser.error('Model identifiers must not be empty')
    manifest = download_models(models, args.manifest, cache_dir=args.cache_dir, refresh=args.refresh,
                               max_workers=args.max_workers, progress=args.progress, log_interval=args.log_interval)
    print(json.dumps({'manifest': str(Path(args.manifest).resolve()),
                      'models': {model: entry['revision'] for model, entry in manifest['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
