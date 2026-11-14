"""Command-line entry points; heavy model imports remain optional."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .retrieval import DecisionRetriever, RetrievalConfig
from .schema import Corpus, load_queries, read_jsonl


def backend_from_args(args):
    if args.backend == 'smoke':
        from .demo import SmokeBackend
        return SmokeBackend()
    if args.backend == 'hf':
        from .backends.hf import HFLogitBackend
        return HFLogitBackend(args.model or 'Qwen/Qwen3-8B', device=args.device, dtype=args.dtype,
                              max_context_tokens=args.max_context_tokens,
                              max_packed_tokens=args.max_packed_tokens, attention=args.attention)
    if args.backend == 'head':
        from .backends.decision_head import DecisionHeadBackend
        if not args.checkpoint:
            raise ValueError('--checkpoint required for a trained decision head')
        return DecisionHeadBackend(args.checkpoint, device=args.device, dtype=args.dtype,
                                   max_context_tokens=args.max_context_tokens,
                                   max_packed_tokens=args.max_packed_tokens, attention=args.attention)
    from .backends.jev import JevBackend
    return JevBackend(provider=args.provider)


def model_args(p):
    p.add_argument('--backend', choices=['hf','head','jev','smoke'], default='hf')
    p.add_argument('--model')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--attention', default='auto', choices=['auto','flex_attention','eager','sdpa'])
    p.add_argument('--max-context-tokens', type=int, default=32768)
    p.add_argument('--max-packed-tokens', type=int, default=40960)
    p.add_argument('--provider', default='typesafe', choices=['typesafe','openrouter'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='DecisionRAG: local hierarchical retrieval and benchmarks')
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare', help='Download/convert a public benchmark')
    prep.add_argument('dataset', choices=['multidoc2dial','conditionalqa','mtrag','scifact','beir','nq-dapr'])
    prep.add_argument('--output', required=True)
    prep.add_argument('--source-dir')
    prep.add_argument('--split')
    prep.add_argument('--limit', type=int)
    demo = subs.add_parser('demo', help='Generate tiny synthetic smoke data')
    demo.add_argument('--output', default='data/demo')
    bench = subs.add_parser('benchmark', help='Measure retrieval quality and serial end-to-end latency')
    bench.add_argument('--corpus', required=True)
    bench.add_argument('--queries', required=True)
    bench.add_argument('--output', required=True)
    bench.add_argument('--method', choices=['decision','bm25','dense'], default='decision')
    bench.add_argument('--config', help='RetrievalConfig JSON file')
    bench.add_argument('--preset', choices=['generic','multidoc2dial','conditionalqa','mtrag','nq'], default='generic')
    bench.add_argument('--pages', type=int)
    bench.add_argument('--top-k', type=int, default=20)
    bench.add_argument('--choice-k', type=int, default=0)
    bench.add_argument('--max-request-tokens', type=int, default=30000)
    bench.add_argument('--max-candidates', type=int, default=10000)
    bench.add_argument('--concurrency', type=int, default=1)
    bench.add_argument('--no-memory', action='store_true')
    bench.add_argument('--independent', action='store_true', help='Ablation: one candidate per context')
    bench.add_argument('--limit', type=int)
    bench.add_argument('--warmup', type=int, default=2)
    bench.add_argument('--dedup-text', action='store_true')
    bench.add_argument('--index-dir')
    bench.add_argument('--batch-size', type=int, default=16)
    model_args(bench)
    records = subs.add_parser('training-data', help='Generate local supervision from training-only queries')
    records.add_argument('--corpus', required=True)
    records.add_argument('--queries', required=True)
    records.add_argument('--output', required=True)
    records.add_argument('--dataset', choices=['multidoc2dial','conditionalqa'], default='multidoc2dial')
    records.add_argument('--tokenizer', default='Qwen/Qwen3-8B-Base')
    records.add_argument('--max-tokens', type=int, default=30000)
    records.add_argument('--max-candidates', type=int, default=128)
    records.add_argument('--seed', type=int, default=42)
    records.add_argument('--limit', type=int)
    compare = subs.add_parser('compare', help='Paired bootstrap confidence interval')
    compare.add_argument('run_a'); compare.add_argument('run_b')
    compare.add_argument('--metric', default='recall@10')
    compare.add_argument('--samples', type=int, default=2000)
    compare.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .datasets import prepare
        result = prepare(args.dataset, args.output, args.source_dir, args.split, args.limit)
    elif args.command == 'demo':
        from .demo import write_demo
        write_demo(args.output)
        result = {'data': args.output, 'warning': 'Synthetic smoke test only; not a quality benchmark'}
    elif args.command == 'compare':
        from .metrics import paired_bootstrap
        result = paired_bootstrap(list(read_jsonl(Path(args.run_a)/'queries.jsonl')),
                                  list(read_jsonl(Path(args.run_b)/'queries.jsonl')),
                                  args.metric, args.samples, args.seed)
    elif args.command == 'training-data':
        from transformers import AutoTokenizer

        from .training_data import make_training_data
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        class TokenCounter:
            def count_tokens(self, text):
                return len(tokenizer.encode(text, add_special_tokens=False))
        result = make_training_data(Corpus.load(args.corpus), load_queries(args.queries), args.output,
                                    TokenCounter(), dataset=args.dataset, seed=args.seed,
                                    max_tokens=args.max_tokens, max_candidates=args.max_candidates,
                                    max_queries=args.limit,
                                    monitor_fraction=.1 if args.dataset == 'conditionalqa' else .03)
    else:
        from .evaluation import evaluate
        corpus = Corpus.load(args.corpus)
        queries = load_queries(args.queries)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError('--limit must be positive')
            queries = queries[:args.limit]
        if not queries:
            raise ValueError('No evaluation queries')
        if args.method == 'decision':
            if args.config:
                cfg = RetrievalConfig(**json.loads(Path(args.config).read_text()))
            else:
                changes = {'top_k': args.top_k, 'choice_k': args.choice_k,
                               'max_request_tokens': args.max_request_tokens, 'max_candidates': args.max_candidates,
                               'concurrency': args.concurrency, 'use_memory': not args.no_memory,
                               'shared_context': not args.independent}
                if args.pages:
                    changes['page_budget'] = args.pages
                    changes['budgets'] = [args.pages]
                cfg = RetrievalConfig.for_dataset(args.preset, **changes)
            backend = backend_from_args(args)
            retriever = DecisionRetriever(corpus, backend, cfg)
            config = {'method': args.method, 'backend': backend.name, 'retrieval': asdict(cfg),
                      'args': vars(args)}
        elif args.method == 'bm25':
            from .baselines import BM25Retriever
            retriever = BM25Retriever(corpus, args.top_k)
            config = vars(args)
        else:
            from .baselines import DenseRetriever
            retriever = DenseRetriever(corpus, args.model or 'Qwen/Qwen3-Embedding-0.6B', args.top_k,
                                       args.device, args.batch_size, args.index_dir)
            config = vars(args)
        result = evaluate(retriever, queries, args.output, warmup=args.warmup,
                          config=config, source_paths=[args.corpus,args.queries], dedup_text=args.dedup_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
