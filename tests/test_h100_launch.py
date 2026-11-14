"""Check the actual training shell launch without installing dependencies or using GPUs."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('selection,expected', [(None, '1'), ('2', '2')])
def test_training_and_evaluation_inherit_one_selected_gpu(tmp_path, selection, expected):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('train_h100.sh', 'gpu_env.sh'):
        shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
    activation = tmp_path / '.venv' / 'bin' / 'activate'
    activation.parent.mkdir(parents=True)
    # Replace heavyweight commands with shell functions while running the real launch script.
    activation.write_text('''
python() { printf '%s|python %s\\n' "$CUDA_VISIBLE_DEVICES" "$*" >> "$COMMAND_LOG"; }
decision-rag() { printf '%s|decision-rag %s\\n' "$CUDA_VISIBLE_DEVICES" "$*" >> "$COMMAND_LOG"; }
''')
    log = tmp_path / 'commands.log'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,2', 'COMMAND_LOG': str(log), 'RUN_TAG': 'test'}
    env.pop('DECISION_RAG_GPU_INDEX', None)
    if selection is not None:
        env['DECISION_RAG_GPU_INDEX'] = selection
    subprocess.run(['bash', str(scripts / 'train_h100.sh'), 'multidoc2dial', 'smoke'],
                   check=True, env=env, capture_output=True, text=True)
    calls = [line.split('|', 1) for line in log.read_text().splitlines()]
    assert {mask for mask, command in calls} == {expected}
    assert any(command.startswith('python -m decision_rag.training ') for _, command in calls)
    evaluation = next(command for _, command in calls if command.startswith('decision-rag benchmark '))
    assert f'--gpu-monitor-index {expected} ' in evaluation
