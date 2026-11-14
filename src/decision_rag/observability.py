"""Per-run logs and terminal progress, without touching model timing or stdout."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm


def _positive_interval(value):
    try:
        interval = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Log interval must be a positive number') from exc
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError('Log interval must be finite and positive')
    return interval


def add_logging_args(parser):
    parser.add_argument('--progress', choices=['auto', 'on', 'off'],
                        default=os.getenv('DECISION_RAG_PROGRESS', 'auto'),
                        help='Progress bars: auto for terminals, on to force, off for plain logs')
    parser.add_argument('--log-interval', type=_positive_interval,
                        default=os.getenv('DECISION_RAG_LOG_INTERVAL', '30'),
                        help='Seconds between progress log records (default: 30)')


class RunReporter:
    """Append flushed human/JSONL logs; render bars only to stderr.

    Callers own output-directory overwrite checks. Each context has a distinct
    run_id so a failed initialization retry can append without erasing evidence.
    """

    def __init__(self, output_dir, *, progress='auto', log_interval=30.0, name='run'):
        if progress not in {'auto', 'on', 'off'}:
            raise ValueError('progress must be auto, on or off')
        self.log_interval = _positive_interval(log_interval)
        self.output = Path(output_dir)
        self.name, self.progress_mode = name, progress
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self._log = self._events = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self._log = (self.output / 'run.log').open('a', encoding='utf-8')
        try:
            self._events = (self.output / 'events.jsonl').open('a', encoding='utf-8')
            self.event('run_started', f'{self.name}: logs in {self.output}',
                       cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                       progress_mode=self.progress_mode)
        except BaseException:
            self._log.close()
            if self._events:
                self._events.close()
            raise
        return self

    def event(self, event, message=None, *, level='INFO', console=True, **fields):
        if self._events is None or self._events.closed:
            raise RuntimeError('Use RunReporter as a context manager')
        timestamp = datetime.now(UTC).isoformat(timespec='milliseconds')
        row = {**fields, 'timestamp': timestamp, 'run_id': self.run_id, 'name': self.name,
               'elapsed_seconds': time.monotonic() - self.started,
               'event': event, 'level': level, 'message': message or event}
        self._events.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self._events.flush()
        visible_fields = {key: value for key, value in fields.items() if key != 'traceback'}
        details = json.dumps(visible_fields, ensure_ascii=False, default=str) if visible_fields else ''
        line = f'{timestamp} {level} [{self.name}] {message or event}'
        if details:
            line += ' ' + details
        self._log.write(line + '\n')
        if fields.get('traceback'):
            self._log.write(fields['traceback'].rstrip() + '\n')
        self._log.flush()
        if console:
            tqdm.write(line, file=sys.stderr)

    def phase(self, description, *, total=None, unit='it', **fields):
        return _Phase(self, description, total=total, unit=unit, fields=fields)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is None:
                self.event('run_completed', f'{self.name}: completed')
            else:
                self.event('run_failed', f'{self.name}: {type(exc).__name__}: {exc}', level='ERROR',
                           error_type=type(exc).__name__, error=str(exc),
                           traceback=''.join(traceback.format_exception(exc_type, exc, tb)))
        finally:
            self._events.close()
            self._log.close()
        return False


class _Phase:
    def __init__(self, reporter, description, *, total, unit, fields):
        if total is not None and total < 0:
            raise ValueError('Progress total must be nonnegative')
        self.reporter, self.description = reporter, description
        self.total, self.unit, self.fields = total, unit, fields
        self.completed, self.metrics = 0, {}
        self.bar = None

    def __enter__(self):
        self.started = self.last_log = time.monotonic()
        self.reporter.event('phase_started', self.description, phase=self.description,
                            total=self.total, unit=self.unit, **self.fields)
        enabled = self.reporter.progress_mode == 'on' or (
            self.reporter.progress_mode == 'auto' and sys.stderr.isatty())
        self.bar = tqdm(total=self.total, desc=self.description, unit=self.unit, file=sys.stderr,
                        disable=not enabled, dynamic_ncols=True, mininterval=.5, leave=False)
        return self

    def set_metrics(self, **metrics):
        self.metrics.update(metrics)
        self.bar.set_postfix(self.metrics, refresh=False)

    def update(self, n=1, **metrics):
        self.completed += n
        self.set_metrics(**metrics)
        self.bar.update(n)
        now = time.monotonic()
        if now - self.last_log >= self.reporter.log_interval:
            self._report('progress', now)
            self.last_log = now

    def _report(self, event, now, **extra):
        elapsed = max(now - self.started, 0.0)
        rate = self.completed / elapsed if elapsed > 0 else None
        eta = max(self.total - self.completed, 0) / rate if self.total is not None and rate else None
        count = f'{self.completed}/{self.total}' if self.total is not None else str(self.completed)
        self.reporter.event(event, f'{self.description}: {count} {self.unit}',
                            phase=self.description, completed=self.completed, total=self.total,
                            unit=self.unit, phase_elapsed_seconds=elapsed, rate_per_second=rate,
                            eta_seconds=eta, metrics=dict(self.metrics), **self.fields, **extra)

    def __exit__(self, exc_type, exc, tb):
        self.bar.close()
        self._report('phase_completed' if exc is None else 'phase_failed', time.monotonic(),
                     **({'level': 'ERROR', 'error': str(exc)} if exc is not None else {}))
        return False
