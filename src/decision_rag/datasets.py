"""Deterministic adapters for public retrieval datasets, without remote Python execution.

Source formats are documented in docs/datasets.md. Downloaded data are never committed.
MD2D grouping/grounding specification: IBM/multidoc2dial, scripts/data_preprocessor.py.
CQA h3/h4 inclusion is an explicit reconstruction from the original data and paper counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .schema import Corpus, Node, Query, read_jsonl, write_jsonl

MD2D_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
CQA_BASE = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/77bd295952daf415548b3244db10880d3d55cfe0/v1_0"
MTRAG_BASE = "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/e0970de1b1207c2a9506adf238e6e1a22037e38a"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DOMAINS = {
    "ssa": "Social Security: retirement, disability, survivors, benefits and eligibility.",
    "va": "Veterans Affairs: veterans' health care, education, disability and benefits.",
    "dmv": "Department of Motor Vehicles: driving licences, vehicle registration and insurance.",
    "studentaid": "Federal Student Aid: student loans, grants, repayment and financial aid.",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _download(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "decision-rag/0.1 research"})
    try:
        with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _unzip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as data:
        for info in data.infolist():
            out = (target / info.filename).resolve()
            if not out.is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if info.filename.startswith("__MACOSX/"):
                continue
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                with data.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def _find(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.exists():
        return direct
    found = sorted(p for p in directory.rglob(name) if "__MACOSX" not in p.parts)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {name} under {directory}, found {len(found)}")
    return found[0]


def _line(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def _normalized(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", text)).split()).lower()


def _md2d_segments(document: dict) -> list[tuple[str, list[str]]]:
    """Official consecutive same-section OR same-heading span grouping, preserving source order."""
    groups: list[list[dict]] = []
    for span in document["spans"].values():
        previous = groups[-1][-1] if groups else None
        if previous is not None and (
            span["id_sec"] == previous["id_sec"] or span["title"].strip() == previous["title"].strip()
        ):
            groups[-1].append(span)
        else:
            groups.append([span])
    results = []
    for group in groups:
        last = group[-1]
        parents = [document["title"]]
        if len(last["parent_titles"]) > 1:
            parents = [p["text"] for p in last["parent_titles"]]
        path = " / ".join(p.replace("/", "-").rsplit("#", 1)[0] for p in parents)
        text = _line(path + " // " + " ".join(s["text_sp"] for s in group))
        results.append((text, [str(s["id_sp"]) for s in group]))
    return results


def _md2d_grounding(grounding: str, passages: list[Node]) -> list[str]:
    # Official evaluation accepts whitespace-insensitive containment after trimming
    # three boundary characters. Prefer any single containing passage, then a span.
    exact = _line(grounding)
    compact = re.sub(r"\s", "", exact.lower())[3:-3]
    multi: list[str] = []
    for start in range(len(passages)):
        combined = ""
        for end in range(start, len(passages)):
            combined += passages[end].text
            match = exact in combined or (bool(compact) and compact in re.sub(r"\s", "", combined.lower()))
            if match:
                candidate = [p.id for p in passages[start : end + 1]]
                if len(candidate) == 1:
                    return candidate
                multi = candidate
                break
    return multi


def multidoc2dial(source: Path, split: str) -> tuple[Corpus, list[Query], list[Path], dict]:
    if split not in {"train", "validation"}:
        raise ValueError("MultiDoc2Dial supports train/validation; its released test is a dummy split")
    doc_file = _find(source, "multidoc2dial_doc.json")
    query_file = _find(source, f"multidoc2dial_dial_{split}.json")
    raw_docs = _json(doc_file)["doc_data"]
    nodes = [Node("root", "Government guidance", "root")]
    page_documents: dict[tuple[str, str], list[dict]] = defaultdict(list)
    document_lookup = {}
    for domain, documents in raw_docs.items():
        nodes.append(
            Node(f"domain:{domain}", DOMAINS.get(domain, domain), "group", "root", domain, len(nodes))
        )
        for doc_id, doc in documents.items():
            document_lookup[doc_id] = doc
            page_documents[(domain, re.sub(r"#\d+$", "", doc["title"]))].append(doc)
    page_ids = {key: f"page:{i}" for i, key in enumerate(page_documents)}
    for (domain, title), docs in page_documents.items():
        heads = list(
            dict.fromkeys(
                s["title"].strip()
                for d in docs
                for s in d["spans"].values()
                if s["title"].strip() and s["title"].strip() != title
            )
        )
        nodes.append(
            Node(
                page_ids[(domain, title)],
                title + ("\nSections: " + "; ".join(heads) if heads else ""),
                "page",
                f"domain:{domain}",
                title,
                len(nodes),
                {"source_doc_ids": [d["doc_id"] for d in docs]},
            )
        )
    passage_number = 0
    by_document: dict[str, list[Node]] = defaultdict(list)
    for domain, documents in raw_docs.items():
        for doc_id, doc in documents.items():
            page_id = page_ids[(domain, re.sub(r"#\d+$", "", doc["title"]))]
            for text, spans in _md2d_segments(doc):
                node = Node(
                    str(passage_number),
                    text,
                    "passage",
                    page_id,
                    order=passage_number,
                    metadata={"source_doc_id": doc_id, "source_span_ids": spans},
                )
                nodes.append(node)
                by_document[doc_id].append(node)
                passage_number += 1
    queries = []
    unmatched = []
    for domain, dialogues in _json(query_file)["dial_data"].items():
        for dialogue in dialogues:
            history = []
            turns = dialogue["turns"]
            for index, turn in enumerate(turns):
                message = {
                    "role": "assistant" if turn["role"] == "agent" else "user",
                    "content": _line(turn["utterance"]),
                }
                target = turns[index + 1] if index + 1 < len(turns) else None
                eligible = (
                    turn["role"] == "user"
                    and target
                    and target["role"] == "agent"
                    and target["da"] != "respond_no_solution"
                )
                if eligible and turn.get("references") and target.get("references"):
                    # Match the published evaluation's grounding from the next agent turn.
                    doc_id = turn["references"][0]["doc_id"]
                    document = document_lookup[doc_id]
                    refs = [document["spans"][str(r["id_sp"])] for r in target["references"]]
                    grounding = document["doc_text"][
                        min(r["start_sp"] for r in refs) : max(r["end_sp"] for r in refs)
                    ]
                    if message["content"] and _line(grounding):
                        qid = f"{dialogue['dial_id']}_{turn['turn_id']}"
                        gold = _md2d_grounding(grounding, by_document[doc_id])
                        if not gold:
                            unmatched.append(qid)
                        queries.append(
                            Query(
                                qid,
                                message["content"],
                                dict.fromkeys(gold, 1.0),
                                dialogue["dial_id"],
                                int(turn["turn_id"]),
                                list(history),
                                [target["utterance"]],
                                {
                                    "dataset": "multidoc2dial",
                                    "split": split,
                                    "gold_page_ids": [
                                        page_ids[(domain, re.sub(r"#\d+$", "", document["title"]))]
                                    ],
                                },
                            )
                        )
                history.append(message)
    if unmatched:
        raise ValueError(
            f"Cannot map {len(unmatched)} MultiDoc2Dial grounding spans; first IDs: {unmatched[:5]}"
        )
    return (
        Corpus(nodes),
        queries,
        [doc_file, query_file],
        {"segmentation": "official consecutive section/heading spans"},
    )


def conditionalqa(source: Path, split: str) -> tuple[Corpus, list[Query], list[Path], dict]:
    if split not in {"train", "dev"}:
        raise ValueError("ConditionalQA supports labeled train/dev splits")
    doc_file, query_file = _find(source, "documents.json"), _find(source, f"{split}.json")
    nodes = [Node("root", "UK government guidance", "root")]
    evidence_lookup = {}
    heading_count = 0
    for index, doc in enumerate(_json(doc_file)):
        page_id = f"P{index}"
        part, section = "", ""
        top_headings, passages = [], []
        exact, normalized = defaultdict(list), defaultdict(list)
        for element_index, raw in enumerate(doc["contents"]):
            match = re.match(r"^\s*<(\w+)(?:\s[^>]*)?>(.*)</\1>\s*$", raw, re.DOTALL)
            tag, text = (match[1].lower(), match[2].strip()) if match else ("raw", raw.strip())
            pid = f"{page_id}:e{element_index}"
            if tag == "h1":
                part, section = text, ""
                top_headings.append(text)
            elif tag == "h2":
                section = text
            else:
                # The release includes h3/h4 as retrievable units. Excluding those
                # would produce 56,127 instead of the paper's 58,483 units.
                passages.append(
                    Node(
                        pid,
                        text,
                        "passage",
                        page_id,
                        order=len(nodes) + element_index,
                        metadata={
                            "heading_path": [x for x in (doc["title"], part, section) if x],
                            "element_index": element_index,
                            "html_tag": tag,
                        },
                    )
                )
            value = None if tag in {"h1", "h2"} else pid
            exact[raw.strip()].append(value)
            normalized[_normalized(text)].append(value)
        heads = list(dict.fromkeys(h for h in top_headings if h and h != doc["title"]))
        nodes.append(
            Node(
                page_id,
                doc["title"] + ("\nSections: " + "; ".join(heads) if heads else ""),
                "page",
                "root",
                doc["title"],
                len(nodes),
                {"url": doc["url"]},
            )
        )
        nodes.extend(passages)
        evidence_lookup[doc["url"]] = (page_id, exact, normalized)
    queries, unmatched = [], []
    for row in _json(query_file):
        page_id, exact, normalized = evidence_lookup[row["url"]]
        gold = set()
        for evidence in row.get("evidences") or []:
            hits = exact.get(evidence.strip()) or normalized.get(_normalized(evidence))
            if hits is None:
                unmatched.append((row["id"], evidence))
            else:
                gold.update(h for h in hits if h is not None)
                heading_count += sum(h is None for h in hits)
        text = f"User's situation: {row['scenario'].strip()}\nUser's question: {row['question'].strip()}"
        queries.append(
            Query(
                str(row["id"]),
                text,
                {pid: 1.0 for pid in sorted(gold)},
                answers=[a[0] for a in row.get("answers", [])],
                metadata={
                    "dataset": "conditionalqa",
                    "split": split,
                    "gold_page_ids": [page_id],
                    "not_answerable": bool(row.get("not_answerable")),
                    "conditional_answers": row.get("answers", []),
                },
            )
        )
    if unmatched:
        raise ValueError(f"Unmapped ConditionalQA evidence: {unmatched[:3]}")
    return (
        Corpus(nodes),
        queries,
        [doc_file, query_file],
        {
            "heading_evidence_excluded": heading_count,
            "segmentation": "all original elements except h1/h2; h3/h4 retained as explicit count-matching reconstruction",
        },
    )


def _qrels(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for index, row in enumerate(csv.reader(stream, delimiter="\t")):
            if not row:
                continue
            if index == 0 and row[0] in {"query-id", "query_id"}:
                continue
            qid, pid, score = row[:3]
            output[qid][pid] = float(score)
    return output


def beir(source: Path, split: str) -> tuple[Corpus, list[Query], list[Path], dict]:
    corpus_file, query_file = source / "corpus.jsonl", source / "queries.jsonl"
    qrel_file = source / "qrels" / f"{split}.tsv"
    nodes = [Node("root", "Document collection", "root")]
    for index, row in enumerate(read_jsonl(corpus_file)):
        pid = str(row["_id"])
        page_id = "beir-page:" + pid
        title = row.get("title", "") or row["text"][:160]
        nodes.extend(
            [
                Node(page_id, title, "page", "root", title, index),
                Node(pid, row["text"], "passage", page_id, order=index),
            ]
        )
    qrels = _qrels(qrel_file)
    queries = [
        Query(str(q["_id"]), q["text"], qrels[str(q["_id"])], metadata={"dataset": "beir", "split": split})
        for q in read_jsonl(query_file)
        if str(q["_id"]) in qrels
    ]
    return (
        Corpus(nodes),
        queries,
        [corpus_file, query_file, qrel_file],
        {
            "segmentation": "one original BEIR document per page and passage; no synthetic passage splitting",
            "comparison": "BEIR transfer benchmark, not one of the paper's four datasets",
        },
    )


def dapr(source: Path, split: str) -> tuple[Corpus, list[Query], list[Path], dict]:
    corpus_file, query_file = source / "corpus.jsonl", source / f"{split}.jsonl"
    nodes = [Node("root", "Wikipedia documents", "root")]
    for index, row in enumerate(read_jsonl(corpus_file)):
        page_id = "dapr-page:" + str(row["id"])
        title = row.get("title") or str(row["id"])
        allowed = row.get("candidate_chunk_ids")
        allowed = None if allowed is None else set(map(str, allowed))
        chunks = [c for c in row["chunks"] if allowed is None or str(c["id"]) in allowed]
        if not chunks:
            continue
        nodes.append(Node(page_id, title, "page", "root", title, index, {"source_doc_id": str(row["id"])}))
        nodes.extend(Node(str(c["id"]), c["text"], "passage", page_id, order=j) for j, c in enumerate(chunks))
    queries = [
        Query(
            str(row["query"]["id"]),
            row["query"]["text"],
            {str(c["chunk"]["id"]): float(c["judgement"]) for c in row["judged_chunks"]},
            metadata={"dataset": "nq-dapr", "split": split},
        )
        for row in read_jsonl(query_file)
    ]
    return (
        Corpus(nodes),
        queries,
        [corpus_file, query_file],
        {
            "segmentation": "DAPR original document chunks",
            "hierarchy": "page/passage; run cluster builder before NQ navigation",
            "comparison": "full selected split; author's unpublished random 500-query subset is not assumed",
        },
    )


def _headings(text: str) -> list[str]:
    lines = text.splitlines()
    result = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not 3 <= len(line) <= 100 or line[-1] in ".:,;!?":
            continue
        if re.match(
            r"^(?:[-*|!\[`]|https?://|\d+[.)]\s|ibmcloud |kubectl |curl |export |docker |pip |git )", line
        ):
            continue
        if (i == 0 or not lines[i - 1].strip()) and (i + 1 == len(lines) or not lines[i + 1].strip()):
            result.append(line)
    return list(dict.fromkeys(result))


def mtrag(source: Path, split: str) -> tuple[Corpus, list[Query], list[Path], dict]:
    if split not in {"human", "dev"}:
        raise ValueError("This MTRAG adapter supports the held-out human IBM Cloud split only")
    passages_file = source / "corpora/passage_level/cloud.jsonl"
    docs_file = source / "corpora/document_level/cloud.jsonl"
    queries_file = source / "mtrag-human/generation_tasks/reference.jsonl"
    qrels_file = source / "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv"
    documents = {str(d["document_id"]): d["text"] for d in read_jsonl(docs_file)}
    pages: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(passages_file):
        match = re.fullmatch(r"(ibmcld_\d+)-(\d+)-(\d+)", row["id"])
        if not match:
            raise ValueError(f"Unexpected IBM Cloud passage ID: {row['id']}")
        row["offset"] = int(match[2])
        pages[match[1]].append(row)
    nodes = [Node("root", "IBM Cloud documentation", "root")]
    products = {}
    for index, (page_id, passages) in enumerate(pages.items()):
        passages.sort(key=lambda p: p["offset"])
        url = urllib.parse.urlparse(passages[0]["url"])
        product_match = re.match(r"/docs/([^/]+)", url.path)
        product = product_match[1] if product_match else url.path.strip("/") or url.netloc
        if product not in products:
            products[product] = Node(
                "product:" + product, product.replace("-", " "), "group", "root", product, index
            )
            nodes.append(products[product])
        text = documents.get(page_id, "")
        heads = _headings(text)
        title = next(
            (
                x.strip()
                for x in text.splitlines()
                if 3 <= len(x.strip()) <= 150 and not re.match(r"^[-*\[!|]|https?://", x.strip())
            ),
            page_id,
        )
        heads = [h for h in heads if h != title]
        nodes.append(
            Node(
                page_id,
                title + ("\nSections: " + "; ".join(heads[:12]) if heads else ""),
                "page",
                "product:" + product,
                title,
                index,
                {"url": passages[0]["url"], "product": product, "headings": heads},
            )
        )
        section = title
        sections = {}
        for p in passages:
            contained = [h for h in heads if _normalized(h) in _normalized(p["text"])]
            if contained:
                section = contained[0]
            if section not in sections:
                section_id = f"{page_id}:section:{len(sections)}"
                sections[section] = section_id
                nodes.append(Node(section_id, section, "section", page_id, section, p["offset"]))
            nodes.append(
                Node(
                    p["id"],
                    p["text"],
                    "passage",
                    sections[section],
                    order=p["offset"],
                    metadata={"heading_path": [title, section], "url": p["url"]},
                )
            )
            if contained:
                section = contained[-1]
    qrels, queries = _qrels(qrels_file), []
    for row in read_jsonl(queries_file):
        if row.get("Collection") not in {"mt-rag-ibmcloud-elser-512-100-20240502", "ibmcloud"}:
            continue
        messages = [
            {"role": "user" if m["speaker"] == "user" else "assistant", "content": m["text"]}
            for m in row["input"]
        ]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"Expected user-final MTRAG input: {row['task_id']}")
        queries.append(
            Query(
                str(row["task_id"]),
                messages[-1]["content"],
                qrels.get(str(row["task_id"]), {}),
                str(row["conversation_id"]),
                int(row["turn"]),
                messages[:-1],
                [t["text"] for t in row.get("targets", [])],
                {"dataset": "mtrag", "split": "human", "relevance": "official gold only"},
            )
        )
    queries.sort(key=lambda q: (q.conversation_id, q.turn_id))
    return (
        Corpus(nodes),
        queries,
        [passages_file, docs_file, queries_file, qrels_file],
        {
            "relevance": "official gold only; paper's expanded GPT-5 pooled judgments are not included",
            "headings": "deterministic layout heuristic inferred from paper; exact heading extraction unspecified",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    dataset: str,
    output_dir: str | Path,
    source_dir: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> dict:
    """Write corpus.jsonl, queries.jsonl and manifest.json. Limit never shrinks corpus."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    dataset = {"md2d": "multidoc2dial", "cqa": "conditionalqa", "nq": "nq-dapr"}.get(dataset, dataset)
    defaults = {
        "multidoc2dial": "validation",
        "conditionalqa": "dev",
        "scifact": "test",
        "beir": "test",
        "mtrag": "human",
        "nq-dapr": "test",
    }
    if dataset not in defaults:
        raise ValueError(f"Unsupported dataset: {dataset}; choose {', '.join(defaults)}")
    split = split or defaults[dataset]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source_dir) if source_dir else output / "raw"
    if source_dir is None:
        if dataset == "multidoc2dial":
            _unzip(_download(MD2D_URL, source / "multidoc2dial.zip"), source)
        elif dataset == "conditionalqa":
            for name in ("documents.json", f"{split}.json"):
                _download(f"{CQA_BASE}/{name}", source / name)
        elif dataset == "scifact":
            _unzip(_download(SCIFACT_URL, source / "scifact.zip"), source)
            source = source / "scifact"
        elif dataset == "mtrag":
            for level in ("document_level", "passage_level"):
                path = f"corpora/{level}/cloud.jsonl.zip"
                archive = _download(f"{MTRAG_BASE}/{path}", source / path)
                _unzip(archive, archive.parent)
            for path in (
                "mtrag-human/generation_tasks/reference.jsonl",
                "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv",
            ):
                _download(f"{MTRAG_BASE}/{path}", source / path)
        else:
            raise ValueError(
                f"{dataset} requires --source-dir with the original dataset export; see docs/datasets.md"
            )
    adapter = {
        "multidoc2dial": multidoc2dial,
        "conditionalqa": conditionalqa,
        "scifact": beir,
        "beir": beir,
        "mtrag": mtrag,
        "nq-dapr": dapr,
    }[dataset]
    corpus, all_queries, files, details = adapter(source, split)
    passage_ids = {p.id for p in corpus.passages}
    for query in all_queries:
        missing = set(query.relevant_ids) - passage_ids
        if missing:
            raise ValueError(f"Query {query.id} references missing passages: {sorted(missing)[:5]}")
    if len({q.id for q in all_queries}) != len(all_queries):
        raise ValueError("Duplicate query IDs")
    queries = all_queries if limit is None else all_queries[:limit]
    corpus.save(output / "corpus.jsonl")
    write_jsonl(output / "queries.jsonl", (asdict(q) for q in queries))
    manifest = {
        "dataset": dataset,
        "split": split,
        "queries": len(queries),
        "queries_before_limit": len(all_queries),
        "labeled_queries": sum(any(g > 0 for g in q.relevant_ids.values()) for q in queries),
        "pages": len(corpus.pages),
        "passages": len(corpus.passages),
        "nodes": len(corpus.nodes),
        "query_limit": limit,
        "selection": "source-order prefix; full corpus retained",
        "sources": [{"name": str(p.relative_to(source)), "sha256": _sha256(p)} for p in files],
        **details,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=["multidoc2dial", "conditionalqa", "scifact", "beir", "mtrag", "nq-dapr"]
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    print(
        json.dumps(prepare(args.dataset, args.output_dir, args.source_dir, args.split, args.limit), indent=2)
    )


if __name__ == "__main__":
    main()
