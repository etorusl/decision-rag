"""Optional device-level sampling for a local inference server in another process."""
from __future__ import annotations

import subprocess
import threading


class GPUMonitor:
    def __init__(self, device_index=None, interval=.25):
        self.device_index, self.interval = device_index, interval
        self.samples, self.peak, self.total_utilization = 0, 0, 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '-i', str(self.device_index),
                 '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, stderr=subprocess.PIPE, timeout=5,
            )
            memory, utilization = [float(x.strip()) for x in output.strip().split(',')]
            self.peak = max(self.peak, int(memory * 2**20))
            self.total_utilization += utilization
            self.samples += 1
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.error = str(exc)
            self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        if self.device_index is not None:
            self._sample()
            if not self.error:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
        if self.device_index is not None and not self.error:
            self._sample()

    def summary(self):
        return {
            'enabled': self.device_index is not None, 'device_index': self.device_index,
            'scope': 'Sampled entire GPU including every process and vLLM KV reservation; not allocator peak',
            'sampling_interval_seconds': self.interval, 'samples': self.samples,
            'peak_device_used_bytes': self.peak if self.samples else None,
            'mean_utilization_percent': self.total_utilization / self.samples if self.samples else None,
            'error': self.error,
        }
