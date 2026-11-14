# Готовые модели: локальный vLLM, только inference

Основной benchmarking-путь использует **vLLM 0.11.2** и готовые Hugging Face checkpoints:
Qwen3-8B, Qwen3-Embedding-8B, Qwen3-Reranker-8B. Обучение, LoRA и подготовка training data
для этих запусков не нужны. Версия зафиксирована по проверенным API; это не требование
использовать «самую новую» версию vLLM.

## Запуск на одной H100 80 GB

```bash
git pull
bash scripts/setup_vllm_h100.sh
bash scripts/benchmark_h100.sh multidoc2dial 100
```

По умолчанию скрипты используют физическую **GPU 1**: серверы vLLM получают
`CUDA_VISIBLE_DEVICES=1`, а мониторинг опрашивает GPU 1. Внутри сервера выбранная карта
видна как `cuda:0`. Общий параметр скриптов — `DECISION_RAG_GPU_INDEX` (по умолчанию `1`);
для конкретного benchmark-запуска его можно явно переопределить через `--gpu-index N`.

Нужны Linux, Python 3.12 и NVIDIA driver, совместимый с CUDA 12.8. Создаётся отдельное
окружение `.venv-vllm`: vLLM 0.11.2 использует PyTorch 2.9.0, а training-окружение проекта
фиксирует 2.7.1. Не устанавливайте `constraints-h100.txt` или `.[local]` в `.venv-vllm`.
Установка не загружает model weights; они скачиваются при запуске серверов.

Скрипт автоматически:

1. Подготавливает полный корпус и выбранное число evaluation queries.
2. Фиксирует Hugging Face revision SHA для каждой модели и запускает сервер с этим SHA.
3. Измеряет BM25 на CPU.
4. Запускает embedding server на `127.0.0.1:8001` с бюджетом 40% VRAM; измеряет dense.
5. Запускает reranker server на `127.0.0.1:8002` с ещё 40%; измеряет live dense → reranker.
6. Останавливает оба своих сервера и запускает Qwen3 на `127.0.0.1:8000` с 85% VRAM.
7. Измеряет DecisionRAG без обучения, с binary scoring и с дополнительным choice reranking.
8. Останавливает свой сервер и сохраняет CSV/Markdown-сводку.

Скрипт не завершает чужие процессы: занятый порт приводит к явной ошибке. Логи и точные
команды серверов находятся в `results/RUN/DATASET/servers/`, revisions — в `server-config.json`.
Повторный run tag не перезаписывается. Для другой группы портов добавьте `--port-base 8100`.

```bash
bash scripts/benchmark_h100.sh multidoc2dial all
bash scripts/benchmark_h100.sh conditionalqa all
bash scripts/benchmark_h100.sh scifact all
bash scripts/benchmark_h100.sh mtrag all

# Быстрая диагностика на готовых 0.6B checkpoints
NAIVE_MODEL=Qwen/Qwen3-0.6B \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B \
bash scripts/benchmark_h100.sh multidoc2dial 20

# Бюджеты страниц, размер shared context и память диалога
bash scripts/ablate_h100.sh multidoc2dial 100
```

`100` ограничивает запросы, не корпус. Для запросов длиннее pooling limit скрипт завершится
ошибкой без скрытого truncation; можно повысить `--pool-max-model-len 16384` после проверки
памяти. Основная модель имеет `--max-model-len 32768`; локальная разбивка кандидатов учитывает
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

## Что именно измеряется

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
- **Work:** logical input tokens, cached tokens (если сервер сообщает), completion tokens,
  HTTP requests и model sequences. Количество внутренних GPU forward passes неизвестно и
  сохраняется как `null`, а не выдумывается из количества запросов.
- **Память:** `nvidia-smi` опрашивается каждые 250 ms. Это sampled device memory целого GPU,
  включая все процессы и KV reservation. Не `torch.cuda.max_memory_allocated` клиентского процесса.
  Короткие пики между samples могут быть пропущены. `--gpu-monitor-index` задаёт физический
  индекс для уже запущенного сервера. В runner `--gpu-index` одновременно выбирает
  физическую GPU для серверов через `CUDA_VISIBLE_DEVICES` и для мониторинга; default — `1`.

Сохраняются timings, запросы, параметры, версии, hashes, server logs и model revisions.
Сравнивайте одни и те же corpus/query SHA и precision. Для повторов задавайте разные `RUN_TAG`.

## Прогресс и логи

В терминале автоматически появляются progress bars прогрева и запросов: выполнено/всего,
скорость, elapsed/ETA и текущие метрики. Runner сообщает, какой из пяти бенчмарков
(семи для абляций) выполняется, и показывает ожидание запуска vLLM с путём к server log.
Общая ETA для разных методов не вычисляется: их скорость различается.

В `results/<RUN_TAG>/<dataset>/` сохраняются `run.log` и `events.jsonl` всего прогона;
в каталоге каждого метода — собственные `run.log`, `events.jsonl`, `queries.jsonl` и `summary.json`.
Traceback при ошибке остаётся в логах. stdout/stderr серверов vLLM находятся в `servers/*.log`.
Логирование и обновление progress bar выполняются вне измеряемого retrieval-интервала.

`--progress auto` — значение по умолчанию; `on` принудительно включает bars,
`off` оставляет обычные строки. Без TTY (например, `nohup` или перенаправление в файл)
режим `auto` выводит обычные строки прогресса каждые 30 секунд, а начала/концы этапов — сразу.
Интервал задаёт `--log-interval`; те же параметры доступны у `decision-rag benchmark`.

```bash
bash scripts/benchmark_h100.sh multidoc2dial 100 --progress auto --log-interval 10

# Запуск после выхода из SSH; каталог results создайте до перенаправления:
mkdir -p results
RUN_TAG=md2d-full nohup bash scripts/benchmark_h100.sh multidoc2dial all \
  --progress off --log-interval 30 > results/md2d-full-console.log 2>&1 &
tail -f results/md2d-full-console.log
```

Общие настройки CLI можно задать через `DECISION_RAG_PROGRESS` и `DECISION_RAG_LOG_INTERVAL`.

CUDA/vLLM inference на H100 пока не запускался из этой CPU-среды; HTTP-контракты и маршрутизация
проверяются тестами, фактическую скорость покажет серверный прогон.

Официальные источники pinned API: [vLLM 0.11.2 models](https://docs.vllm.ai/en/v0.11.2/models/supported_models/),
[OpenAI-compatible server](https://docs.vllm.ai/en/v0.11.2/serving/openai_compatible_server/),
[Qwen3 reranker example](https://github.com/vllm-project/vllm/blob/v0.11.2/examples/offline_inference/pooling/qwen3_reranker.py),
[GPU installation](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/).
