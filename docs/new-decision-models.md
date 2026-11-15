# Qwen3.5-9B и Gemma-4-12B-it после текущего прогона

Последовательность для H100 80 GB, только **физическая GPU 1**:

1. Дождаться завершения `naive-binary-vllm` текущего Qwen3-8B и остановить suite.
2. Скачать `Qwen/Qwen3.5-9B` и `google/gemma-4-12B-it`.
3. Проверить **одни и те же 20 запросов** в binary и choice для каждой новой модели.
4. Досчитать Qwen3-8B naive choice из исходного прогона.
5. Выполнить binary и choice на прежней полной выборке для Qwen3.5, затем Gemma.

Используется подтверждённая instruction-tuned Gemma с суффиксом `-it`.
`google/gemma-4-12B` — базовый checkpoint без chat template; подмена между ними не выполняется.
Для этих экспериментов используются только текстовые входы и однопозиционный scoring
готовых моделей. Обучение и генерация цепочки рассуждений не нужны.

## Окружения

| Прогоны | Окружение | vLLM | Transformers | PyTorch / CUDA |
| --- | --- | --- | --- | --- |
| Прежние Qwen3/BGE | `.venv-vllm` | 0.11.2 | 4.57.1 | 2.9.0 / 12.8 |
| Qwen3.5 и Gemma4 Unified | `.venv-vllm-modern` | 0.24.0+cu129 | 5.10.4 | 2.11.0 / 12.9 |

Старое окружение и training `.venv` остаются отдельными. Modern setup проверяет CUDA
и BF16 на выбранной GPU; драйвер должен поддерживать этот CUDA runtime. Окружение
фиксирует версии, доступные для обеих архитектур; это не утверждение об оптимальной
версии vLLM по производительности. Версия **сервера** сохраняется в итоговом отчёте.
Установщик использует отдельный release wheel `0.24.0+cu129` с проверкой SHA256:
обычная сборка `vllm==0.24.0` рассчитана на CUDA 13 и здесь не используется.

## Команды на сервере

После завершения binary и остановки текущего suite:

```bash
cd ~/decision-rag
git pull --ff-only
export DECISION_RAG_GPU_INDEX=1
export CUDA_VISIBLE_DEVICES=1

bash scripts/setup_vllm_modern_h100.sh

.venv-vllm-modern/bin/python scripts/download_models.py \
  --models Qwen/Qwen3.5-9B google/gemma-4-12B-it \
  --manifest cache/new-decision-models.json
```

Этот отдельный manifest фиксирует SHA обеих новых моделей и не меняет manifest прежнего
Qwen3/BGE набора. Повтор скачивания сохраняет SHA, пока явно не передан `--refresh`.
При запуске vLLM получает проверенный локальный `snapshot_path` из manifest, а API
и отчёты сохраняют исходный ID модели и SHA. Это позволяет использовать скачанные
файлы для инференса offline: отсутствие README/LICENSE или альтернативных форматов
весов в кеше не требует скачивания всего Hugging Face репозитория.

Задайте исходный каталог и новый каталог эксперимента. Сохраните значение
`new_models_run`, если переключаетесь между терминалами.

```bash
source_bench_run=results/20261114T192438644288Z/multidoc2dial
new_models_run="results/qwen35-gemma4-$(date -u +%Y%m%dT%H%M%SZ)"

.venv-vllm-modern/bin/python scripts/run_decision_model_experiments.py \
  "$source_bench_run" --stage smoke \
  --experiment-dir "$new_models_run" \
  --model-manifest cache/new-decision-models.json \
  --query-concurrency 8 --gpu-index 1
```

Smoke означает 20 уникальных evaluation-запросов на модель, каждый проходит оба метода
(binary и choice). Warmup выполняется отдельно и не входит в эти 20. Это проверка
загрузки модели, tokenizer/chat template, single-token labels/logprobs и полного
retrieval-пайплайна; 20 запросов не заменяют итоговую оценку качества.

Если smoke завершился ошибкой, после исправления причины повторите ту же команду
с тем же `--experiment-dir`. Скрипт проверит неизменность данных, manifest и настроек,
пропустит успешно завершённые модели и повторит неудачный прогон модели. Его прежние
логи и частичные результаты сохраняются в `failed-attempts/` внутри эксперимента.
Оба метода неудачной модели запускаются заново; завершённый smoke другой модели
не повторяется. Повторный запуск уже успешного smoke только проверяет результаты.
Одновременные smoke-запуски в одной папке и состояния `smoke_running` после аварийного
завершения процесса отклоняются. При ошибке старта сервера конец его лога выводится
в терминал; полный лог остаётся в `servers/decision.log`.

После успешного smoke:

```bash
.venv-vllm-modern/bin/python scripts/run_decision_model_experiments.py \
  "$source_bench_run" --stage full \
  --experiment-dir "$new_models_run" \
  --query-concurrency 8 --gpu-index 1
```

`full` запускает по очереди:

- Qwen3-8B **choice**, через старый `.venv-vllm`, с прежним SHA модели;
- Qwen3.5-9B **binary и choice**, через `.venv-vllm-modern`;
- Gemma-4-12B-it **binary и choice**, через `.venv-vllm-modern`.

Если исходный choice уже завершён, он пропускается. Binary исходного Qwen3-8B должен
иметь завершённые `summary.json` и `queries.jsonl` именно в указанном исходном каталоге;
иначе `full` остановится до запуска моделей. Если binary досчитывался в другом recovery-run,
нужно указать каталог с его фактическими результатами и подходящей конфигурацией.

Полный query limit, корпус, запросы и retrieval context budgets наследуются из исходного
прогона. `full` перепроверяет smoke-результаты, SHA моделей и хеши данных; он не стартует
после незавершённого smoke или незаметной смены manifest. Нейросетевые модели запускаются
последовательно, workers разделяют один vLLM-сервер выбранной модели.

Старый каталог не изменяется. Новые результаты, `experiment.json`, логи и общий отчёт
находятся внутри `new_models_run`. Smoke-строки не смешиваются с полными прогонами
в итоговом сравнении. Каждая строка содержит модель, режим нагрузки и concurrency;
отдельно показаны фактическая версия сервера, wall QPS и latency под нагрузкой.

## Модельные настройки и пределы проверки

Обе новые модели запускаются с `--language-model-only`, BF16 и TP=1. Qwen3.5 использует
гибридную attention/DeltaNet архитектуру; для её prefix caching явно включены
`--mamba-cache-mode align` и chunked prefill. Попадания в prefix cache зависят от границ
блоков; priming не гарантирует полного повторного использования префикса.
Спекулятивный MTP не включается: для нашего однопозиционного scoring он не нужен.

Chat template берётся из зафиксированного checkpoint с `enable_thinking=False`.
Клиент проверяет соответствие локального и серверного tokenizer и валидность
нормализованных label probabilities. Ошибки не переключают инференс на другую модель
или backend. На локальной машине без GPU проверяются orchestration и протокол;
реальный CUDA smoke выполняется приведённой командой на H100.

Первичные источники: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it),
[vLLM 0.24 GPU installation](https://docs.vllm.ai/en/v0.24.0/getting_started/installation/gpu/),
[vLLM 0.24 release wheels](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.24.0),
[vLLM 0.24 model registry](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/registry.py).
