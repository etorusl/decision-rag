# Полный запуск Qwen + BGE на H100 80 GB

Все команды ниже выполняются **на удалённом Linux-сервере**, из корня репозитория.
Нужны Python 3.12 и CUDA-совместимый драйвер для окружения vLLM из setup-скрипта.
Используется только физическая **GPU 1**: внутри CUDA-процессов она видна как `cuda:0`.
Это бенчмарк готовых моделей через локальный vLLM; обучение для этих прогонов не требуется.

## 1. Репозиторий и окружение

Для нового клона:

```bash
git clone git@github.com:etorusl/decision-rag.git
cd decision-rag
```

Для существующего клона:

```bash
cd ~/decision-rag
git pull --ff-only
```

Затем:

```bash
export DECISION_RAG_GPU_INDEX=1
export CUDA_VISIBLE_DEVICES=1
bash scripts/setup_vllm_h100.sh
```

Скрипт создаёт `.venv-vllm`, проверяет CUDA и устанавливает vLLM 0.11.2.
Окружение обучения `.venv` существует отдельно. Все последующие команды используют
инференс-окружение явно, поэтому активировать его вручную не нужно.

## 2. Скачать пять моделей до бенчмарка

```bash
.venv-vllm/bin/python scripts/download_models.py \
  --manifest cache/model-manifest.json
```

Будут скачаны:

- `Qwen/Qwen3-8B` — naive binary и naive choice;
- `Qwen/Qwen3-Embedding-8B` — прежний dense;
- `Qwen/Qwen3-Reranker-8B` — прежний dense + rerank;
- `BAAI/bge-m3` — дополнительный dense;
- `BAAI/bge-reranker-v2-m3` — дополнительный reranker.

Загрузчик выбирает один формат весов и сохраняет точные commit SHA, пути и список файлов
в `cache/model-manifest.json`. Повтор этой команды использует сохранённые SHA и докачивает
недостающие файлы. Для намеренного обновления моделей добавьте `--refresh`.
Логи загрузки находятся в `cache/model-download/`.

Если HF-кэш нужно разместить на отдельном диске, в первую команду добавьте, например,
`--cache-dir /mnt/models/huggingface/hub`. Runner возьмёт путь из manifest автоматически.
Manifest нужно создавать на сервере, где будут запуски: он содержит абсолютные пути.

При передаче `--model-manifest` runner проверяет локальные snapshots до запуска и включает
offline-режим Hugging Face для серверов и модельных клиентов. Подготовка датасетов всё ещё
может обращаться в интернет. Данные и индексы сохраняются локально между запусками.

## 3. Пробный прогон: 100 запросов, все семь вариантов

```bash
bash scripts/benchmark_h100.sh multidoc2dial 100 \
  --model-manifest cache/model-manifest.json \
  --execution-mode throughput --query-concurrency 8
```

| Каталог прогона | Метод | Модель / embedder | Reranker |
| --- | --- | --- | --- |
| `bm25` | BM25 | — | — |
| `dense-vllm` | dense | `Qwen/Qwen3-Embedding-8B` | — |
| `dense-rerank-vllm` | dense + rerank | `Qwen/Qwen3-Embedding-8B` | `Qwen/Qwen3-Reranker-8B` |
| `dense-bge-vllm` | dense | `BAAI/bge-m3` | — |
| `dense-rerank-bge-vllm` | dense + rerank | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` |
| `naive-binary-vllm` | naive binary | `Qwen/Qwen3-8B` | — |
| `naive-choice-vllm` | naive choice | `Qwen/Qwen3-8B` | — |

Прежние пять вариантов сохраняют модели, retrieval budgets и имена каталогов. `--skip-bge`
позволяет запустить только их; прежний последовательный профиль доступен через
`--execution-mode latency --query-concurrency 1`. BGE использует dense-векторы; sparse и ColBERT-выходы BGE-M3 здесь
не участвуют. BGE получает обычный запрос и документ без Qwen-инструкций.

GPU выполняет нейросетевой инференс, dense dot products/top-k и численную часть BM25.
Текстовая подготовка, токенизация, файловый ввод-вывод и orchestration выполняются на CPU.
Qwen и BGE pooling-серверы запускаются последовательно; внутри одной пары embedding-сервер
и reranker могут работать одновременно. После pooling-прогонов они останавливаются, затем
запускается `Qwen3-8B`. Не запускайте несколько suites одновременно на GPU 1 для сравнения времени.

Общие настройки: top-20 результатов, 100 кандидатов для rerank, batch size 16.
Прогрев: два запроса на каждый логический worker в throughput и два всего в latency
(на короткой выборке число ограничено её длиной). Фактическое число записывается в summary;
warmup не входит в wall-clock QPS. GPU-индекс при этом остаётся общим.
Dense-поиск — точный FP32 на GPU. Pooling context limit — 8192
для обеих пар; BGE имеет отдельный параметр `--bge-max-model-len` (не более 8192),
Qwen — `--pool-max-model-len`. Контекст decision-модели — 32768.

Runner по умолчанию использует **throughput с максимум восемью активными запросами**
для каждого из семи методов. Это начальный профиль для H100, а не измеренный оптимум.
Новый запрос запускается после завершения предыдущего на освободившемся worker
(closed-loop нагрузка). Независимые диалоги идут параллельно, turns каждого диалога —
строго по порядку. Если независимых диалогов меньше восьми, реальная параллельность ниже.
Веса моделей и GPU-индекс общие; query concurrency не создаёт восемь vLLM-серверов.

Для измерения задержки одного запроса используйте отдельный run tag:

```bash
bash scripts/benchmark_h100.sh multidoc2dial 100 \
  --model-manifest cache/model-manifest.json \
  --execution-mode latency --query-concurrency 1 \
  --run-tag "latency-$(date -u +%Y%m%dT%H%M%SZ)"
```

Throughput-отчёт показывает **latency под нагрузкой** и **wall-clock QPS**: число
завершённых запросов, делённое на длительность измеренного прогона. Это не обратная
величина среднего времени запроса. Изменение batching может изменить относительную
скорость методов; ускорение каждого метода не гарантировано. Сравнивайте одинаковые
execution mode и query concurrency; latency и throughput — отдельные эксперименты.

## 4. Полные прогоны и единый отчёт

Эта команда последовательно проходит все четыре доступных датасета. Три первых —
структурированные retrieval-бенчмарки из экспериментального плана по статье;
SciFact — дополнительная проверка на научных текстах. Подробности выборок и отличий
от статьи: [README](../README.md#бенчмарки).

```bash
bench_tag="qwen-bge-$(date -u +%Y%m%dT%H%M%SZ)"
(
  set -e
  for bench_dataset in multidoc2dial conditionalqa mtrag scifact; do
    bash scripts/benchmark_h100.sh "$bench_dataset" all \
      --model-manifest cache/model-manifest.json \
      --execution-mode throughput --query-concurrency 8 \
      --run-tag "$bench_tag"
  done
  .venv-vllm/bin/python scripts/summarize_runs.py "results/$bench_tag" \
    --output "results/$bench_tag/comparison"
)
```

Результаты:

- `results/<tag>/<dataset>/comparison.md` и `.csv` — сравнение внутри датасета;
- `results/<tag>/comparison.md` и `.csv` — общий отчёт, без усреднения разных датасетов;
- `results/<tag>/<dataset>/<method>/queries.jsonl` и `summary.json` — данные отдельных запросов;
- `results/<tag>/<dataset>/run.log`, `events.jsonl` — ход suite;
- `results/<tag>/<dataset>/servers/` — команды и логи Qwen-серверов;
- `results/<tag>/<dataset>/servers/bge/` — команды и логи BGE-серверов.

В **каждой строке каждой таблицы** отчёта указаны тип прогона и модель. Например:
`naive binary — Qwen/Qwen3-8B` или
`dense + rerank — embedder: BAAI/bge-m3; reranker: BAAI/bge-reranker-v2-m3`.
В подписи сохраняется и каталог датасета, например `[multidoc2dial/dense-bge-vllm]`.
В CSV также есть отдельные поля моделей, revision SHA и inference engine.
Идентичность модели берётся из сохранённой конфигурации конкретного прогона.

Сравнивайте качество и время внутри одного датасета, с одинаковыми SHA входов,
query limit, top-k и candidate budget. Отчёт показывает quality-метрики,
режим и concurrency, latency, wall-clock QPS, доступные token usage и пиковую занятую VRAM по GPU.
Serial QPS относится только к последовательному профилю.
Загрузка весов, построение индекса и warmup не входят в измеренную latency запроса.
Для dense в данных по запросам отдельно сохранены время embedding и GPU-поиска.
Это retrieval-бенчмарк: генерация финального ответа и её качество здесь не измеряются.

## 5. Дополнительные прогоны

### Восстановить прерванный suite с новым concurrency

Обновите код и укажите прежний каталог датасета. Например:

```bash
git pull --ff-only
.venv-vllm/bin/python scripts/resume_vllm_benchmark.py \
  results/OLD_TAG/multidoc2dial \
  --execution-mode throughput --query-concurrency 8
```

`--dry-run` покажет план без запуска серверов и записи файлов. Готовым считается метод
с корректными `summary.json` и полным набором записей `queries.jsonl`. Скрипт проверяет
выборку и хеши текущих данных по завершённым результатам, сохраняет прежние model IDs,
revision SHA, query limit и retrieval context limits. Подготовка датасета повторно не запускается.
Используется прежний model manifest, если он был указан; его новый путь можно передать
через `--model-manifest`. Без manifest точные SHA берутся из старого `server-config.json`.

Завершённые BM25/dense/naive методы пропускаются. Частично выполненный метод запускается
**с начала** в новом каталоге: частичные timings со старым режимом не смешиваются с новым
throughput, а память диалогов строится заново в правильном порядке. Старый каталог,
включая незавершённые запросы, остаётся нетронутым. Если всё уже завершено, серверы не запускаются.
Если нет ни одного готового метода, исторические хеши проверить нельзя; это явно отражено
как `data_hash_validation: unavailable`, и все незавершённые методы запускаются заново.

Новые результаты находятся в `results/<old-tag>-resume-c8-<timestamp>/<dataset>/`.
Внутри создаются `recovery-plan.json` и общий `recovery-comparison.md/.csv`, который объединяет
только проверенные старые результаты и новые завершённые методы. Старые latency и новые
throughput строки подписаны отдельно: их время относится к разным режимам нагрузки.

Для ручного запуска только выбранных методов в новом suite:

```bash
bash scripts/benchmark_h100.sh multidoc2dial all \
  --methods naive-binary-vllm naive-choice-vllm \
  --model-manifest cache/model-manifest.json \
  --execution-mode throughput --query-concurrency 8
```

Runner поднимет только decision-сервер. Он также поддерживает `--reuse-revisions PATH`
для точных SHA из старого `server-config.json` и `--skip-prepare` для готовых локальных данных.
При совместном использовании manifest и старых SHA несовпадение версий считается ошибкой.

### Подбор concurrency и другие проверки

Подбор query concurrency на одной и той же выборке; suites запускаются последовательно,
каждая настройка получает собственный tag. Scheduler limits и retrieval budgets остаются
одинаковыми во всех пяти throughput-прогонах, включая concurrency 1:

```bash
sweep_tag="h100-concurrency-$(date -u +%Y%m%dT%H%M%SZ)"
(
  set -e
  for query_concurrency in 1 2 4 8 16; do
    bash scripts/benchmark_h100.sh multidoc2dial 100 \
      --model-manifest cache/model-manifest.json \
      --execution-mode throughput --query-concurrency "$query_concurrency" \
      --run-tag "${sweep_tag}-q${query_concurrency}"
  done
)
```

Повторите перспективные настройки, проверьте quality-метрики и сравните QPS с p95 latency.
Большая занятая VRAM сама по себе не означает высокую загрузку вычислительных блоков GPU.
Настройки scheduler и клиентских batches описаны в [vLLM](vllm.md#параллельные-запросы-и-batching).

Повтор полного прогона на одном датасете (новый tag создастся автоматически):

```bash
bash scripts/benchmark_h100.sh conditionalqa all \
  --model-manifest cache/model-manifest.json
```

Абляции naive Decision RAG: разные page/context budgets и отключение памяти.
Для них используется только уже скачанный `Qwen/Qwen3-8B`; dense-пары не запускаются:

```bash
bash scripts/ablate_h100.sh multidoc2dial all \
  --model-manifest cache/model-manifest.json
```

Повторно построить отчёт по выбранному существующему tag:

```bash
.venv-vllm/bin/python scripts/summarize_runs.py "results/$bench_tag" \
  --output "results/$bench_tag/comparison"
```

В терминале progress bars включаются автоматически. Для запуска через `nohup` удобнее
`--progress off --log-interval 30`: сохранятся обычные строки прогресса и файлы логов.
Подробности метрик и настроек: [vLLM](vllm.md), [retrieval](vllm-retrieval.md).
