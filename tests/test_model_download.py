"""Prefetch pins, file selection and atomic manifests without downloading weights."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'model_download', Path(__file__).resolve().parents[1] / 'scripts/download_models.py')
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def siblings(*names):
    return [SimpleNamespace(rfilename=name, size=10) for name in names]


def test_prefetch_selects_one_weight_format_and_retains_tokenizer_and_pooling_config():
    files, fmt = download.model_files(siblings(
        'config.json', 'tokenizer.json', 'sentencepiece.bpe.model', '1_Pooling/config.json',
        'model.safetensors', 'pytorch_model.bin', 'pytorch_model.bin.index.json',
        'onnx/model.onnx', 'onnx/config.json', 'colbert_linear.pt'))
    assert fmt == 'safetensors'
    assert set(files) == {'config.json', 'tokenizer.json', 'sentencepiece.bpe.model',
                          '1_Pooling/config.json', 'model.safetensors'}
    files, fmt = download.model_files(siblings('config.json', 'pytorch_model.bin', 'tokenizer_config.json'))
    assert fmt == 'pytorch' and 'pytorch_model.bin' in files


@pytest.mark.parametrize('names', [
    # Official Qwen3.5-9B file layout: the shard names include "model.safetensors-".
    ['config.json', 'chat_template.jinja', 'merges.txt', 'vocab.json', 'tokenizer.json',
     'tokenizer_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json',
     'model.safetensors.index.json',
     *[f'model.safetensors-{index:05d}-of-00004.safetensors' for index in range(1, 5)]],
    # Gemma 4 keeps its text/tokenizer and multimodal processor metadata in JSON.
    ['config.json', 'generation_config.json', 'processor_config.json', 'tokenizer.json',
     'tokenizer_config.json', 'model.safetensors'],
])
def test_new_decision_model_layouts_retain_all_weights_templates_and_processor_metadata(names):
    files, fmt = download.model_files(siblings(*names, 'README.md', 'onnx/model.onnx'))
    assert fmt == 'safetensors'
    assert set(files) == set(names)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    state = {'main': 'a' * 40, 'info': [], 'downloads': [], 'fail': None, 'omit': None}
    class Api:
        def model_info(self, model, revision=None, files_metadata=False):
            assert files_metadata
            state['info'].append((model, revision))
            return SimpleNamespace(sha=revision or state['main'], siblings=siblings(
                'config.json', 'tokenizer.json', 'model.safetensors', 'pytorch_model.bin'))
    def snapshot_download(**kwargs):
        state['downloads'].append(kwargs)
        if kwargs['repo_id'] == state['fail']:
            raise OSError('interrupted download')
        path = Path(kwargs['cache_dir']) / ('models--' + kwargs['repo_id'].replace('/', '--')) / 'snapshots' / kwargs['revision']
        path.mkdir(parents=True, exist_ok=True)
        for name in kwargs['allow_patterns']:
            if name != state['omit']:
                (path / name).write_text('fixture')
        return str(path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=Api, constants=SimpleNamespace(HF_HUB_CACHE=str(tmp_path / 'hub')),
        snapshot_download=snapshot_download))
    return state


def test_downloads_immutable_snapshots_and_reuses_pins_until_explicit_refresh(fake_hub, tmp_path):
    path = tmp_path / 'manifest.json'
    first = download.download_models(['org/model', 'org/model'], path, progress='off')
    assert len(fake_hub['downloads']) == 1
    assert fake_hub['downloads'][0]['revision'] == 'a' * 40
    assert 'pytorch_model.bin' not in fake_hub['downloads'][0]['allow_patterns']
    assert first['models']['org/model']['size_bytes'] == 30
    assert Path(first['models']['org/model']['snapshot_path']).is_dir()
    fake_hub['main'] = 'b' * 40
    second = download.download_models(['org/model'], path, progress='off')
    assert fake_hub['info'][-1] == ('org/model', 'a' * 40)
    assert second['models']['org/model']['revision'] == 'a' * 40
    third = download.download_models(['org/model'], path, refresh=True, progress='off')
    assert fake_hub['info'][-1] == ('org/model', None)
    assert third['models']['org/model']['revision'] == 'b' * 40
    assert json.loads(path.read_text()) == third


def test_failed_download_keeps_existing_manifest_intact(fake_hub, tmp_path):
    path = tmp_path / 'manifest.json'
    download.download_models(['org/model'], path, progress='off')
    original = path.read_bytes()
    fake_hub['fail'] = 'org/other'
    with pytest.raises(OSError, match='interrupted'):
        download.download_models(['org/model', 'org/other'], path, progress='off')
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('manifest.json.*.tmp'))


def test_incomplete_cache_does_not_produce_manifest(fake_hub, tmp_path):
    fake_hub['omit'] = 'model.safetensors'
    path = tmp_path / 'manifest.json'
    with pytest.raises(RuntimeError, match='Incomplete'):
        download.download_models(['org/model'], path, progress='off')
    assert not path.exists()


def test_default_download_command_includes_all_five_models(monkeypatch, capsys):
    calls = []
    def fake(models, *args, **kwargs):
        calls.extend(models)
        return {'models': {model: {'revision': 'a' * 40} for model in models}}
    monkeypatch.setattr(download, 'download_models', fake)
    for name in ('NAIVE_MODEL', 'EMBEDDING_MODEL', 'RERANKER_MODEL'):
        monkeypatch.delenv(name, raising=False)
    download.main([])
    assert calls == ['Qwen/Qwen3-8B', 'Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B',
                     'BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3']
    assert set(json.loads(capsys.readouterr().out)['models']) == set(calls)


def test_explicit_model_list_uses_its_own_manifest_and_ignores_suite_environment(fake_hub, tmp_path, monkeypatch, capsys):
    baseline = tmp_path / 'baseline-manifest.json'
    download.download_models(['Qwen/Qwen3-8B'], baseline, progress='off')
    original = baseline.read_bytes()
    fake_hub['info'].clear()
    fake_hub['downloads'].clear()
    monkeypatch.setenv('NAIVE_MODEL', 'unrequested/naive-model')
    monkeypatch.setenv('EMBEDDING_MODEL', 'unrequested/embedding-model')
    monkeypatch.setenv('RERANKER_MODEL', 'unrequested/reranker-model')
    manifest = tmp_path / 'new-decision-models.json'
    models = ['Qwen/Qwen3.5-9B', 'google/gemma-4-12B']
    command = ['--models', *models, '--manifest', str(manifest), '--progress', 'off']
    download.main(command)
    assert [model for model, _ in fake_hub['info']] == models
    assert [call['repo_id'] for call in fake_hub['downloads']] == models
    saved = json.loads(manifest.read_text())
    assert list(saved['models']) == models
    assert baseline.read_bytes() == original
    result = json.loads(capsys.readouterr().out)
    assert result['manifest'] == str(manifest.resolve())
    assert result['models'] == dict.fromkeys(models, 'a' * 40)
    fake_hub['main'] = 'b' * 40
    download.main(command)
    assert fake_hub['info'][-2:] == [(model, 'a' * 40) for model in models]
    assert baseline.read_bytes() == original


@pytest.mark.parametrize('conflicting', [
    ['--skip-bge'], ['--naive-model', 'org/naive'], ['--embedding-model', 'org/embed'],
    ['--reranker-model', 'org/rerank'], ['--bge-embedding-model', 'org/bge-embed'],
    ['--bge-reranker-model', 'org/bge-rerank'],
])
def test_explicit_model_list_rejects_ambiguous_suite_options_before_downloading(monkeypatch, conflicting, capsys):
    monkeypatch.setattr(download, 'download_models', lambda *args, **kwargs: pytest.fail('must not download'))
    with pytest.raises(SystemExit) as exc:
        download.main(['--models', 'Qwen/Qwen3.5-9B', *conflicting])
    assert exc.value.code == 2
    assert '--models cannot be combined' in capsys.readouterr().err


@pytest.mark.parametrize('models', [[], [''], ['   ']])
def test_explicit_model_list_requires_nonempty_ids(monkeypatch, models):
    monkeypatch.setattr(download, 'download_models', lambda *args, **kwargs: pytest.fail('must not download'))
    with pytest.raises(SystemExit) as exc:
        download.main(['--models', *models])
    assert exc.value.code == 2


def test_default_role_options_and_environment_still_apply_without_explicit_list(monkeypatch, capsys):
    calls = []
    def fake(models, *args, **kwargs):
        calls.extend(models)
        return {'models': {model: {'revision': 'a' * 40} for model in models}}
    monkeypatch.setattr(download, 'download_models', fake)
    monkeypatch.setenv('NAIVE_MODEL', 'custom/naive')
    monkeypatch.setenv('EMBEDDING_MODEL', 'custom/embedding-from-env')
    monkeypatch.setenv('RERANKER_MODEL', 'custom/reranker')
    download.main(['--embedding-model', 'custom/embedding-override', '--skip-bge'])
    assert calls == ['custom/naive', 'custom/embedding-override', 'custom/reranker']
    assert set(json.loads(capsys.readouterr().out)['models']) == set(calls)
