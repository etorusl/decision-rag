"""Backport the upstream batch-mask fix to the pinned vLLM 0.24 wheel."""
from importlib.metadata import distribution
from pathlib import Path
from tempfile import NamedTemporaryFile

SWAP_FIX = 'vllm-project/vllm#48419'
WORKER_SOURCE = 'vllm/v1/worker/gpu_input_batch.py'
UNFIXED_SWAP = '''        if self.allowed_token_ids_mask_cpu_tensor is not None:
            (
                self.allowed_token_ids_mask_cpu_tensor[i1],
                self.allowed_token_ids_mask_cpu_tensor[i2],
            ) = (
                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],
            )
'''
FIXED_SWAP = '''        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[[i1, i2]] = (
                self.allowed_token_ids_mask_cpu_tensor[[i2, i1]]
            )
'''
REPAIR_COMMAND = '.venv-vllm-modern/bin/python -m decision_rag.vllm_compat'


def _worker_path(installed):
    return Path(installed.locate_file(WORKER_SOURCE))


def require_batch_mask_fix(installed):
    source = _worker_path(installed).read_text(encoding='utf-8')
    if source.count(FIXED_SWAP) != 1 or UNFIXED_SWAP in source:
        raise RuntimeError(f'vLLM 0.24.0 needs the batch-mask fix {SWAP_FIX}. Run {REPAIR_COMMAND}')


def apply_batch_mask_fix(installed):
    if installed.version.split('+', 1)[0] != '0.24.0':
        raise RuntimeError(f'This backport targets vLLM 0.24.0, found {installed.version}')
    path = _worker_path(installed)
    source = path.read_text(encoding='utf-8')
    if source.count(FIXED_SWAP) == 1 and UNFIXED_SWAP not in source:
        return False
    if source.count(UNFIXED_SWAP) != 1 or FIXED_SWAP in source:
        raise RuntimeError(f'Unexpected vLLM 0.24.0 worker source at {path}. No files changed.')

    # Only replace the known buggy block. Atomic replacement keeps the module
    # readable if setup is interrupted and preserves the wheel's file mode.
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(source.replace(UNFIXED_SWAP, FIXED_SWAP, 1))
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


if __name__ == '__main__':
    changed = apply_batch_mask_fix(distribution('vllm'))
    print(f'vLLM batch-mask fix {SWAP_FIX}: {"applied" if changed else "already applied"}')
