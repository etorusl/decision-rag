import argparse
import io
import json

import pytest

from decision_rag.observability import RunReporter, add_logging_args


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_plain_logs_are_throttled_and_include_real_rate_and_eta(tmp_path, monkeypatch, capsys):
    now = [0.0]
    monkeypatch.setattr('decision_rag.observability.time.monotonic', lambda: now[0])
    with RunReporter(tmp_path, progress='off', log_interval=30) as report, \
            report.phase('Queries', total=4, unit='query') as stage:
        now[0] = 5
        stage.update(1, loss=.3)
        now[0] = 31
        stage.update(1, loss=.2)
        now[0] = 40
        stage.update(2, loss=.1)
    events = rows(tmp_path / 'events.jsonl')
    progress = [row for row in events if row['event'] == 'progress']
    assert len(progress) == 1
    assert progress[0]['completed'] == 2
    assert progress[0]['rate_per_second'] == pytest.approx(2 / 31)
    assert progress[0]['eta_seconds'] == 31
    complete = next(row for row in events if row['event'] == 'phase_completed')
    assert complete['completed'] == complete['total'] == 4
    assert complete['eta_seconds'] == 0
    assert complete['metrics']['loss'] == .1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert '\r' not in captured.err and '\x1b' not in captured.err
    assert 'Queries' in captured.err and 'Queries' in (tmp_path / 'run.log').read_text()


@pytest.mark.parametrize('mode,tty,expected_bar', [('auto', False, False), ('auto', True, True),
                                                ('on', False, True), ('off', True, False)])
def test_terminal_detection_and_explicit_progress_override(tmp_path, monkeypatch, mode, tty, expected_bar):
    class Terminal(io.StringIO):
        def isatty(self):
            return tty
    terminal = Terminal()
    monkeypatch.setattr('decision_rag.observability.sys.stderr', terminal)
    with RunReporter(tmp_path, progress=mode) as report, report.phase('Training', total=1) as stage:
        stage.update(loss=.5)
    assert ('\r' in terminal.getvalue()) is expected_bar
    assert '\r' not in (tmp_path / 'run.log').read_text()


def test_failure_preserves_traceback_and_closes_files(tmp_path):
    report = RunReporter(tmp_path, progress='off')
    with pytest.raises(RuntimeError, match='model failed'), report, report.phase('Loading model'):
        raise RuntimeError('model failed')
    events = rows(tmp_path / 'events.jsonl')
    assert events[-1]['event'] == 'run_failed'
    assert events[-1]['error_type'] == 'RuntimeError'
    assert 'RuntimeError: model failed' in events[-1]['traceback']
    assert 'Traceback (most recent call last)' in (tmp_path / 'run.log').read_text()
    assert all(row['event'] != 'run_completed' for row in events)
    assert report._log.closed and report._events.closed


def test_unknown_work_does_not_invent_eta_and_retries_preserve_logs(tmp_path):
    for _ in range(2):
        with RunReporter(tmp_path, progress='off') as report, report.phase('Waiting for server') as stage:
            stage.update(0)
    events = rows(tmp_path / 'events.jsonl')
    assert len({row['run_id'] for row in events}) == 2
    for row in events:
        if row['event'] == 'phase_completed':
            assert row['total'] is None and row['eta_seconds'] is None


def test_logging_options_take_env_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv('DECISION_RAG_PROGRESS', 'off')
    monkeypatch.setenv('DECISION_RAG_LOG_INTERVAL', '7.5')
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    assert vars(parser.parse_args([])) == {'progress': 'off', 'log_interval': 7.5}
    assert vars(parser.parse_args(['--progress', 'on', '--log-interval', '2'])) == {
        'progress': 'on', 'log_interval': 2.0}


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'bad'])
def test_invalid_log_interval_fails_before_running(value):
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(['--log-interval', value])
