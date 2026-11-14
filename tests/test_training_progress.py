"""Training diagnostics must survive load failures and leave stdout machine-readable."""
import json

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from decision_rag import training


def test_model_load_failure_is_persisted_and_propagated(monkeypatch, tmp_path, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic model load failure")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fail)
    output = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="synthetic model load failure"):
        training.train_records(tmp_path / "train.jsonl", output, model_name="unused-model",
                               device="cpu", progress="off")
    assert capsys.readouterr().out == ""
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "run_failed" for event in events)
    assert not any(event["event"] == "training_completed" for event in events)
    log = (output / "run.log").read_text()
    assert "Traceback" in log
    assert "synthetic model load failure" in log


def test_training_cli_routes_progress_options_without_polluting_json(monkeypatch, capsys):
    calls = []
    def train(*args, **kwargs):
        calls.append((args, kwargs))
        return {"trained_steps": 3}
    monkeypatch.setattr(training, "train_records", train)
    training.main(["--train", "train.jsonl", "--output", "checkpoint",
                   "--progress", "off", "--log-interval", "7"])
    assert json.loads(capsys.readouterr().out) == {"trained_steps": 3}
    assert calls[0][1]["progress"] == "off"
    assert calls[0][1]["log_interval"] == 7
