# DecisionRAG

Самостоятельная реализация **Retriever-Free Retrieval: Navigating Documents with Decision Models**
([arXiv:2610.35868](https://arxiv.org/abs/2610.35868)). Основной сценарий — одна **H100 80 GB**, без внешнего API.

**Бенчмаркинг готовых моделей использует локальный vLLM, только inference.**
Qwen3, Qwen3-Embedding, Qwen3-Reranker и дополнительная пара BGE-M3 запускаются как localhost-серверы на H100.
Обучение собственной head — отдельный эксперимент.

Реализованы два подхода:

1. **Без обучения:** Qwen3-8B оценивает `P(yes | context, question)` по logits. Все кандидаты видны в общем контексте. vLLM оценивает отдельные вопросы с общим prefix KV и максимум одним label token. Дополнительный HF backend использует packed attention. Опционально — multiple-choice reranking.
2. **С обучением:** Qwen3-8B-Base + cosine scoring-head из приложения G, изолированные ветки question/options, LoRA и локальная подготовка supervision. Отдельный checkpoint обязателен.

Это воспроизведение архитектуры с документированными допущениями. **Локальные модели не равны Jev.**
Исходная general-trained модель, точная смесь обучения и ряд параметров не заданы в PDF;
мы не приписываем своей реализации цифры статьи. См. [соответствие статье](docs/paper-fidelity.md)
и [локальные модели](docs/local-models.md).

## Запуск на сервере

Нужны Linux, Python 3.12, H100 80 GB и NVIDIA driver с поддержкой CUDA 12.8.
Модели и данные скачиваются на сервер; API-ключи не нужны.
Скрипты vLLM и обучения по умолчанию используют физическую **GPU 1** через
`CUDA_VISIBLE_DEVICES=1`; внутри процесса это `cuda:0`.

```bash
git clone git@github.com:etorusl/decision-rag.git
cd decision-rag
bash scripts/setup_vllm_h100.sh
source .venv-vllm/bin/activate
python scripts/download_models.py --manifest cache/model-manifest.json

# Семь систем, первые 100 запросов, полный корпус:
# BM25 / Qwen dense и dense+reranker / BGE dense и dense+reranker / naive binary и choice
bash scripts/benchmark_h100.sh multidoc2dial 100 --model-manifest cache/model-manifest.json

# Полные бенчмарки без обучения
bash scripts/benchmark_h100.sh multidoc2dial all --model-manifest cache/model-manifest.json
bash scripts/benchmark_h100.sh conditionalqa all --model-manifest cache/model-manifest.json
```

Сначала запускайте короткую выборку. Скрипт сам поднимает и останавливает локальные vLLM-серверы;
для готовых моделей training-этапа нет. Установка vLLM изолирована в `.venv-vllm`.
Предварительное скачивание фиксирует SHA и проверяет файлы всех пяти моделей. С manifest
модельные процессы используют локальный HF cache без сети; данные скачиваются отдельно.
Исходные Qwen-прогоны сохранены, к ним добавлены `BAAI/bge-m3` и
`BAAI/bge-reranker-v2-m3`. В каждой строке отчёта указаны тип прогона и фактические модели.
Dense-поиск и top-k по умолчанию выполняются на GPU вместе с embedding/reranker;
FP32-индекс переносится туда один раз при инициализации. Устройство поиска и время
query embedding / vector search записываются в результаты; CPU-поиск требует явного `--search-device cpu`.
BM25 scoring и top-k также используют GPU; текстовая подготовка и токенизация выполняются на CPU.
Для другой GPU задайте `DECISION_RAG_GPU_INDEX=N`; в benchmark runner также доступен `--gpu-index N`.
H100 memory/latency и качество больших моделей **ещё не измерены** на машине разработки.
Подробные команды, параметры и обработка OOM: [инструкция vLLM](docs/vllm.md).
Полная последовательность скачивания, smoke/full-прогонов и отчёта: [команды запуска](docs/run-benchmarks.md).

## Бенчмарки

| Данные | Что проверяем | Основные метрики | Состояние подготовки |
|---|---|---|---|
| MultiDoc2Dial | диалог, маршрутизация, память страниц | R@1/5/10 | 413 страниц, 4 110 passages, 4 201 validation queries |
| ConditionalQA | сценарии и несколько evidence paragraphs | Hit@1, R@5/10, nDCG | 652 страницы, 58 483 units, 285 dev queries / 271 labeled |
| MTRAG IBM Cloud | большая техническая документация | R@5, nDCG@5 | 8 578 страниц, 72 442 passages, 205 turns / 188 labeled |
| Natural Questions / DAPR | масштаб и topic routing | Hit@1, R@10 | локальный импорт DAPR + построитель 1 000 кластеров |
| SciFact / BEIR | перенос на научные тексты | nDCG@10, Recall@10 | 5 183 abstracts, 300 test claims |
| Synthetic stress | влияние размера и длины корпуса | latency, work, VRAM | генератор; не доказательство качества на реальных задачах |

Рекомендуемый порядок: **MultiDoc2Dial → ConditionalQA → SciFact → MTRAG → NQ**.
Первые четыре подготовлены и проверены по реальным официальным данным на CPU.
NQ требует большого исходного DAPR-корпуса и отдельного offline clustering; см.
[данные](docs/datasets.md), [NQ](docs/natural-questions.md), [дополнительные проверки](docs/benchmarks-extra.md).
MTRAG использует официальные gold labels: расширенные GPT-5 labels из статьи нам не даны.

Для скорости сохраняются p50/p90/p95/p99, serial QPS, пиковая память CUDA,
число HTTP-запросов, logical sequences, tokens и времена этапов. Внутренние vLLM forward passes
не наблюдаются и отмечаются `null`; sampled GPU memory измеряется через `nvidia-smi`. Измерение включает весь retrieval
после загрузки модели/корпуса; общее время инициализации/индексирования и время прогрева записываются отдельно.
Компиляция новых форм запросов может остаться внутри измеряемых запросов.
Каждый запуск пишет `queries.jsonl` и `summary.json`, commit SHA, версии и SHA256 входов.
Результаты и веса не коммитятся.

Обучение и бенчмарки показывают progress bars с количеством шагов/запросов, скоростью и ETA.
В каталоге каждого запуска сохраняются `run.log` и `events.jsonl`; обучение также пишет
`train_log.jsonl`. Без TTY идут обычные строки прогресса. Настройки: `--progress auto|on|off`
и `--log-interval 30`; для shell-скриптов доступны `DECISION_RAG_PROGRESS` и
`DECISION_RAG_LOG_INTERVAL`. Примеры, включая `nohup`: [логи vLLM](docs/vllm.md#прогресс-и-логи).

```bash
# Таблица по завершённым запускам
python scripts/summarize_runs.py results

# Разность качества и paired bootstrap CI с пересэмплированием целых диалогов
decision-rag compare results/RUN/multidoc2dial/naive-binary-vllm \
  results/RUN/multidoc2dial/dense-rerank-vllm --metric recall@1

# Бюджет страниц, размер общего контекста, отключение памяти
bash scripts/ablate_h100.sh multidoc2dial 100 --model-manifest cache/model-manifest.json
```

## Обучение собственной модели отдельно

```bash
bash scripts/setup_h100.sh  # отдельное .venv с PyTorch 2.7.1
bash scripts/train_h100.sh multidoc2dial smoke
bash scripts/train_h100.sh multidoc2dial full
```

Этот путь не нужен для inference готовых checkpoints. Подробности — [H100](docs/h100.md)
и [scoring-head](docs/local-models.md).

## Разработка без GPU

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,data]'
pytest -q                         # torch-тесты пропускаются, если torch отсутствует

decision-rag demo --output data/demo
decision-rag benchmark --corpus data/demo/corpus.jsonl --queries data/demo/queries.jsonl \
  --backend smoke --pages 1 --output results/smoke
```

`smoke` — явно синтетическая заглушка для проверки CLI, **не retrieval-модель**.
Для tiny-model тестов локального backend установите также `.[local]`; веса из сети не требуются.

## Использование своей коллекции

Формат — JSONL. Узлы `root → group/page → section/passage`, стабильные строковые ID,
поля `id`, `text`, `kind`, `parent_id`, `title`, `order`, `metadata`.
Промпты используют текст и структуру; gold labels читаются только оценщиком и генератором training data.

```python
from decision_rag.schema import Corpus, Query
from decision_rag.backends.hf import HFLogitBackend
from decision_rag.retrieval import DecisionRetriever, RetrievalConfig

corpus = Corpus.load("data/my/corpus.jsonl")
retriever = DecisionRetriever(corpus, HFLogitBackend(),
    RetrievalConfig(budgets=[10], max_request_tokens=30000))
result = retriever.retrieve(Query(id="example", text="Your question"))
for passage_id in result.ids[:5]:
    print(corpus.nodes[passage_id].text)
```

Ориентиры по коду: `retrieval.py` — алгоритм и политики четырёх datasets;
`backends/` — модели; `training_data.py` / `training.py` — обучение;
`baselines.py` / `reranking.py` — сравнения; `evaluation.py` — протокол измерения.
