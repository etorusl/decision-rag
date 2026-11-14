# Готовые модели: локальный vLLM, только inference

Основной benchmarking-путь использует **vLLM 0.11.2** и готовые Hugging Face checkpoints:
Qwen3-8B, Qwen3-Embedding-8B, Qwen3-Reranker-8B, BAAI/bge-m3 и BAAI/bge-reranker-v2-m3.
Обучение, LoRA и подготовка training data
для этих запусков не нужны. Версия зафиксирована по проверенным API; это не требование
использовать «самую новую» версию vLLM.

## Запуск на одной H100 80 GB

Полная последовательность команд от установки до итогового отчёта: [run-benchmarks.md](run-benchmarks.md).

```bash
git pull
bash scripts/setup_vllm_h100.sh
.venv-vllm/bin/python scripts/download_models.py --manifest cache/model-manifest.json
bash scripts/benchmark_h100.sh multidoc2dial 100 --model-manifest cache/model-manifest.json
```

По умолчанию скрипты используют физическую **GPU 1**: серверы vLLM получают
`CUDA_VISIBLE_DEVICES=1`, а мониторинг опрашивает GPU 1. Внутри сервера выбранная карта
видна как `cuda:0`. Общий параметр скриптов — `DECISION_RAG_GPU_INDEX` (по умолчанию `1`);
для конкретного benchmark-запуска его можно явно переопределить через `--gpu-index N`.

Нужны Linux, Python 3.12 и NVIDIA driver, совместимый с CUDA 12.8. Создаётся отдельное
окружение `.venv-vllm`: vLLM 0.11.2 использует PyTorch 2.9.0, а training-окружение проекта
фиксирует 2.7.1. Не устанавливайте `constraints-h100.txt` или `.[local]` в `.venv-vllm`.
Установка не загружает model weights. Отдельный `download_models.py` скачивает все пять
checkpoints до запуска эксперимента и записывает immutable revisions, каталог HF cache
и список файлов в manifest. Повторный вызов сохраняет прежние SHA; `--refresh` явно
обновляет их. С `--model-manifest` runner проверяет наличие файлов и запускает модели
с `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`; подготовка датасета остаётся сетевой.
Без manifest сохранён прежний режим разрешения revisions через Hugging Face.

Скрипт автоматически:

1. Подготавливает полный корпус и выбранное число evaluation queries.
2. Использует проверенные Hugging Face revision SHA из manifest и запускает серверы с этими SHA.
3. Измеряет BM25: численный scoring и top-k выполняются на GPU.
4. Запускает embedding server на `127.0.0.1:8001` с бюджетом 40% VRAM; измеряет dense с поиском на GPU.
5. Запускает reranker server на `127.0.0.1:8002` с ещё 40%; измеряет live dense → reranker.
6. Останавливает оба Qwen pooling-сервера; на тех же портах последовательно запускает
   BGE-M3 embedding и BGE reranker, измеряет ещё dense и dense → reranker, затем освобождает их.
7. Запускает Qwen3 на `127.0.0.1:8000` с 85% VRAM и измеряет DecisionRAG без обучения:
   binary scoring и дополнительный choice reranking.
8. Останавливает свой сервер и сохраняет CSV/Markdown-сводку с моделями в каждой строке.

| Прогон | Модель / эмбеддер | Реранкер |
|---|---|---|
| BM25 | BM25 | — |
| Dense | Qwen/Qwen3-Embedding-8B | — |
| Dense + rerank | Qwen/Qwen3-Embedding-8B | Qwen/Qwen3-Reranker-8B |
| Dense | BAAI/bge-m3 | — |
| Dense + rerank | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 |
| Naive binary | Qwen/Qwen3-8B | — |
| Naive choice | Qwen/Qwen3-8B | выбор через ту же Qwen/Qwen3-8B |

Исходные пять вариантов, модели и retrieval budgets сохранены. `--skip-bge` возвращает исходную матрицу;
по умолчанию выполняются все семь. Дополнительные BGE-прогоны используют dense-векторы BGE-M3,
без sparse/ColBERT-гибрида и без смешивания эмбеддеров и реранкеров разных семейств.

## Параллельные запросы и batching

Runner и shell-скрипты по умолчанию используют `--execution-mode throughput --query-concurrency 8`:
несколько независимых retrieval-запросов одновременно подают работу одному vLLM engine.
Все методы используют одинаковый лимит параллельных запросов. Внутри каждого диалога
следующий turn ждёт предыдущий; разные диалоги могут выполняться параллельно. Это closed-loop
нагрузка без заданной частоты поступления: освободившийся worker берёт следующую работу.
Восемь запросов — стартовый профиль, оптимум на H100 ещё не измерен.
В throughput каждый логический worker прогревается двумя запросами; в latency прогрев
содержит два запроса всего. Для более короткой выборки используется её длина.
Фактическое число warmup-запросов записывается в summary и исключается из wall-clock QPS.
Прогрев разных workers использует общий GPU-индекс без повторного создания индекса.

`--execution-mode latency --query-concurrency 1` воспроизводит последовательный профиль
с прежними scheduler limits. Если concurrency не указана, latency выбирает 1 автоматически;
значение выше 1 для этого режима отклоняется. Прямой `decision-rag benchmark` по умолчанию
остаётся в latency/1; throughput нужно включить явно. Семь методов suite запускаются
последовательно, поэтому разные методы не конкурируют друг с другом за GPU 1.

| Параметр runner | Throughput default | Latency default | Назначение |
| --- | --- | --- | --- |
| `--query-concurrency` | 8 | 1 | Максимум активных retrieval-запросов |
| `--decision-max-num-seqs` | 128 | 64 | Sequence cap decision scheduler |
| `--pool-max-num-seqs` | 64 | 16 | Sequence cap каждого pooling scheduler |
| `--decision-max-num-batched-tokens` | 32768 | 32768 | Tokens за scheduler iteration; auto = max(16384, decision context) |
| `--pool-max-num-batched-tokens` | 16384 | 16384 | Auto = max(16384, context конкретного Qwen/BGE сервера) |
| `--vllm-batch-size` | 64 | 64 | Decision sequences в одном HTTP-запросе |
| `--batch-size` | 16 | 16 | Embedding/reranker inputs в одном HTTP-запросе |

Query concurrency, число sequences и HTTP batch size — разные ограничения. Один retrieval
может отправить несколько HTTP-запросов, а один HTTP-запрос содержит несколько sequences.
Runner не увеличивает вложенную concurrency внутри decision scoring. Увеличение scheduler
limit без достаточной клиентской нагрузки само по себе не ускоряет модель. У BGE отключён
chunked prefill, поэтому pooling token budget должен быть не меньше BGE context limit.
Слишком большой batch/лимит может увеличить latency или привести к OOM.

Бюджеты `--decision-memory .85` и `--pool-memory .40` остаются прежними: это доли VRAM,
не проценты вычислительной загрузки. В throughput память модели и GPU-индекс не размножаются
на число запросов. Дополнительные активные запросы требуют рабочей памяти и KV cache.
Все выбранные параметры записываются в конфигурацию; фактические команды серверов — в logs.

Примеры latency/8-way throughput и sweep concurrency 1/2/4/8/16 с отдельными tags:
[полные команды](run-benchmarks.md). При sweep сохраняйте revision, корпус, prompts,
page/candidate budgets, cache policy и precision; после изменения batching проверяйте качество.
Параллельный режим может поменять порядок методов по скорости, но не гарантирует ускорения.

## Артефакты и дополнительные параметры

Скрипт не завершает чужие процессы: занятый порт приводит к явной ошибке. Логи и точные
команды серверов находятся в `results/RUN/DATASET/servers/`, revisions — в `server-config.json`.
BGE server logs находятся в `servers/bge/` и не перезаписывают Qwen logs.
Повторный run tag не перезаписывается. Для другой группы портов добавьте `--port-base 8100`.

```bash
bash scripts/benchmark_h100.sh multidoc2dial all --model-manifest cache/model-manifest.json
bash scripts/benchmark_h100.sh conditionalqa all --model-manifest cache/model-manifest.json
bash scripts/benchmark_h100.sh scifact all --model-manifest cache/model-manifest.json
bash scripts/benchmark_h100.sh mtrag all --model-manifest cache/model-manifest.json

# Необязательная отдельная диагностика на готовых Qwen 0.6B checkpoints
.venv-vllm/bin/python scripts/download_models.py --skip-bge --manifest cache/small-model-manifest.json \
  --naive-model Qwen/Qwen3-0.6B --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --reranker-model Qwen/Qwen3-Reranker-0.6B
NAIVE_MODEL=Qwen/Qwen3-0.6B \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B \
bash scripts/benchmark_h100.sh multidoc2dial 20 --skip-bge --model-manifest cache/small-model-manifest.json

# Бюджеты страниц, размер shared context и память диалога
bash scripts/ablate_h100.sh multidoc2dial 100 --model-manifest cache/model-manifest.json
```

`100` ограничивает запросы, не корпус. Для запросов длиннее pooling limit скрипт завершится
ошибкой без скрытого truncation. Для Qwen можно повысить `--pool-max-model-len 16384` после проверки
памяти; BGE имеет отдельный `--bge-max-model-len 8192`, максимум 8192 для этих checkpoints.
Основная модель имеет `--max-model-len 32768`; локальная разбивка кандидатов учитывает
этот предел. Бюджет VRAM сервера — предварительное резервирование, не измеренный минимум.

## Подключение к своему уже запущенному vLLM

Для Qwen3 decision scoring:

```bash
source .venv-vllm/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --generation-config vllm --enable-prefix-caching \
  --enable-prompt-tokens-details --logprobs-mode processed_logprobs --max-logprobs 26
```

В другом терминале:

```bash
source .venv-vllm/bin/activate
decision-rag prepare multidoc2dial --output data/multidoc2dial
CUDA_VISIBLE_DEVICES=1 decision-rag benchmark \
  --corpus data/multidoc2dial/corpus.jsonl --queries data/multidoc2dial/queries.jsonl \
  --preset multidoc2dial --backend vllm --model Qwen/Qwen3-8B \
  --base-url http://127.0.0.1:8000 --limit 100 --gpu-monitor-index 1 \
  --output results/naive-vllm
```

Backend проверяет served model и tokenizer, считывает обе label probabilities и нормализует
их. При неверном endpoint, несовпадающем tokenizer или недостающих logits он завершается
ошибкой; переключения на HF/PyTorch без уведомления нет. Для точного повторения запуска
зафиксируйте server `--revision` / `--tokenizer-revision` и client `--model-revision` одним SHA.

Команды pooling-серверов и API: [embedding/reranking](vllm-retrieval.md).
Точная схема single-token scoring: [decision backend](vllm-backend.md).
`decision-rag benchmark --method dense` и `--method dense-rerank` по умолчанию также используют
vLLM (`--engine vllm`). Прежняя реализация доступна явно через `--engine hf` / `--backend hf`.
В обоих dense-бэкендах точный поиск и top-k по умолчанию выполняются на GPU
(`--search-device cuda`). FP32-индекс загружается в GPU один раз при инициализации;
при каждом запросе переносится его вектор, обратно возвращаются только top-k IDs/scores.
Это та же физическая GPU 1, которую используют модели. TF32 отключён для поиска,
embedding/reranker продолжают работать в BF16. CPU-поиск доступен явно через
`--search-device cpu` и записывается в конфиг как отдельный эксперимент.
Этот же параметр управляет численным BM25 scoring и top-k; подготовка текста и токенизация
остаются на CPU.

Размер матрицы индекса — `passages × embedding_dimension × 4` байта плюс CUDA context
и рабочая память поиска. При двух pooling-серверах с бюджетами 40% + 40% индекс занимает
часть оставшейся памяти. При OOM запуск завершится с ошибкой; скрытого перехода на CPU нет.
При необходимости уменьшайте `--pool-memory` с учётом памяти моделей. Результаты старой
версии с CPU-поиском следует перезапустить для сравнения времени с GPU DecisionRAG.

## Что именно измеряется

- **Режим нагрузки:** latency измеряет один последовательный запрос; throughput — задержку
  запроса под нагрузкой и wall-clock QPS всего измеренного прогона. Сумма длительностей
  параллельных запросов не равна wall time. Сравнение режимов и concurrency выполняйте отдельно.
- **Весь online retrieval:** построение prompts, токенизация, localhost HTTP, очередь vLLM,
  prefill/one-token scoring, поиск по embeddings и все этапы навигации. Загрузка серверов,
  offline corpus embedding и прогрев исключены из per-query timer.
- **Одна logical sequence на binary question.** Общий контекст одинаков, vLLM переиспользует
  prefix KV. Это не custom packed-attention forward из HF backend. Serving получает максимум
  один label token на решение; полноценные ответы не генерируются.
- **Prefix cache:** default `per-query` назначает свежий `cache_salt` каждому query, включая
  прогрев. Внутри query reuse разрешён. Первый настоящий candidate выполняется перед остальными,
  чтобы заполнить prefix cache; его время и tokens включены. `--prefix-cache-policy warm`
  разрешает межзапросный reuse и является отдельным экспериментом.
- **Embedding/reranker:** pooling servers запускаются с `--no-enable-prefix-caching`, поэтому
  повтор первого query после warmup не получает скрытого ускорения.
  В dense-stage отдельно записываются `query_embedding_seconds` и `vector_search_seconds`;
  второй включает перенос вектора запроса на GPU и получение готового top-k. Время reranking
  записывается отдельным этапом. Построение и перенос индекса относятся к инициализации.
- **Work:** logical input tokens, cached tokens (если сервер сообщает), completion tokens,
  HTTP requests и model sequences. Количество внутренних GPU forward passes неизвестно и
  сохраняется как `null`, а не выдумывается из количества запросов.
- **Память:** `nvidia-smi` опрашивается каждые 250 ms. Это sampled device memory целого GPU,
  включая индекс в клиенте, все процессы vLLM и KV reservation. Не `torch.cuda.max_memory_allocated` клиентского процесса.
  Короткие пики между samples могут быть пропущены. `--gpu-monitor-index` задаёт физический
  индекс для уже запущенного сервера. В runner `--gpu-index` одновременно выбирает
  физическую GPU для серверов через `CUDA_VISIBLE_DEVICES` и для мониторинга; default — `1`.

Сохраняются timings, запросы, параметры, версии, hashes, server logs и model revisions.
Сравнивайте одни и те же corpus/query SHA и precision. Для повторов задавайте разные `RUN_TAG`.
CSV/Markdown-сводка отдельно показывает устройство и точность поиска, размер индекса,
среднее время query embedding / vector search и sampled пиковую память всей GPU.

## Прогресс и логи

В терминале автоматически появляются progress bars прогрева и запросов: выполнено/всего,
скорость, elapsed/ETA и текущие метрики. Runner сообщает, какой из семи бенчмарков
выполняется и какие модели используются, и показывает ожидание запуска vLLM с путём к server log.
Общая ETA для разных методов не вычисляется: их скорость различается.

В `results/<RUN_TAG>/<dataset>/` сохраняются `run.log` и `events.jsonl` всего прогона;
в каталоге каждого метода — собственные `run.log`, `events.jsonl`, `queries.jsonl` и `summary.json`.
Traceback при ошибке остаётся в логах. stdout/stderr серверов vLLM находятся в `servers/*.log`
и `servers/bge/*.log`.
Логирование и обновление progress bar выполняются вне индивидуального retrieval-интервала;
их orchestration overhead входит в wall-clock throughput.

`--progress auto` — значение по умолчанию; `on` принудительно включает bars,
`off` оставляет обычные строки. Без TTY (например, `nohup` или перенаправление в файл)
режим `auto` выводит обычные строки прогресса каждые 30 секунд, а начала/концы этапов — сразу.
Интервал задаёт `--log-interval`; те же параметры доступны у `decision-rag benchmark`.

```bash
bash scripts/benchmark_h100.sh multidoc2dial 100 --model-manifest cache/model-manifest.json \
  --progress auto --log-interval 10

# Запуск после выхода из SSH; каталог results создайте до перенаправления:
mkdir -p results
RUN_TAG=md2d-full nohup bash scripts/benchmark_h100.sh multidoc2dial all \
  --model-manifest cache/model-manifest.json --progress off --log-interval 30 \
  > results/md2d-full-console.log 2>&1 &
tail -f results/md2d-full-console.log
```

Общие настройки CLI можно задать через `DECISION_RAG_PROGRESS` и `DECISION_RAG_LOG_INTERVAL`.

CUDA/vLLM inference на H100 пока не запускался из этой CPU-среды; HTTP-контракты и маршрутизация
проверяются тестами, фактическую скорость покажет серверный прогон.

Официальные источники pinned API: [vLLM 0.11.2 models](https://docs.vllm.ai/en/v0.11.2/models/supported_models/),
[OpenAI-compatible server](https://docs.vllm.ai/en/v0.11.2/serving/openai_compatible_server/),
[Qwen3 reranker example](https://github.com/vllm-project/vllm/blob/v0.11.2/examples/offline_inference/pooling/qwen3_reranker.py),
[GPU installation](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/).
