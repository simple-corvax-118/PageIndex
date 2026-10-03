"""Offline preparation and explicitly opted-in live JEV test for the MGTM03 guide.

Run from the checkout root: python -m examples.jev_retrieval.teacher_guide ...
No course text, generated index or live response should be committed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from pageindex.jev_retrieval import JevNodeSelector
from pageindex.page_index_md import md_to_tree
from pageindex.utils import create_node_mapping, get_node_path

SPEC_PATH = Path(__file__).with_name("teacher_guide_cases.json")
DEFAULT_OUTPUT = Path(".pageindex/jev_teacher_guide")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare(document: Path, output_dir: Path, *, allow_changed_source=False):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    data = document.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["source"]["sha256"] and not allow_changed_source:
        raise ValueError("Source checksum differs from the supplied Drive document. "
                         "Use the original file or explicitly pass --allow-changed-source.")
    # Use the REAL existing Markdown indexer. No summaries, thinning, document
    # description, token counting, PageIndex cloud, or LLM/API calls.
    index = asyncio.run(md_to_tree(
        str(document), if_thinning=False, if_add_node_id="yes",
        if_add_node_summary="no", if_add_doc_description="no", if_add_node_text="yes"))
    mapping = create_node_mapping(index["structure"])
    if not mapping or not all("text" in node for node in mapping.values()):
        raise ValueError("Prepared index has no nodes or is missing node text.")
    write_json(output_dir / "index.json", index)
    manifest = {"source_sha256": digest, "source_size_bytes": len(data),
                "expected_source_match": digest == spec["source"]["sha256"],
                "source_name": document.name, "node_count": len(mapping),
                "index_sha256": hashlib.sha256((output_dir / "index.json").read_bytes()).hexdigest(),
                "indexer": "pageindex.page_index_md.md_to_tree",
                "llm_calls": 0, "index_mode": "markdown_without_summaries_or_thinning"}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def evaluate(output_dir: Path, *, case_id="all", threshold=None, live=False, retain_raw=False):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    cases = [case for case in spec["cases"] if case_id in ("all", case["id"])]
    if not cases:
        raise ValueError(f"Unknown case ID: {case_id}")
    index_bytes = (output_dir / "index.json").read_bytes()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(index_bytes).hexdigest() != manifest["index_sha256"]:
        raise ValueError("Prepared index changed; run prepare again before evaluating.")
    tree = json.loads(index_bytes)["structure"]
    mapping = create_node_mapping(tree)
    selector = JevNodeSelector(config={} if threshold is None else {"jev_retrieval_threshold": threshold})
    rows = []
    # Resolve all expectations before making any paid requests.
    targets = {}
    for case in cases:
        titles = set(case.get("expected_any_sections", []))
        missing = titles - {node["title"] for node in mapping.values()}
        if missing:
            raise ValueError(f"Expected section titles absent from source for {case['id']}: {sorted(missing)}")
        targets[case["id"]] = {node_id for node_id in mapping
                               if any(node["title"] in titles
                                      for node in get_node_path(tree, node_id))}
    for case in cases:
        row = {"case_id": case["id"], "query": case["query"]}
        if not live:
            requests = selector.prepare_requests(case["query"], tree, include_text=True)
            row.update(request_count=len(requests), evaluated_nodes=len(mapping),
                       expected_target_ids=sorted(targets[case["id"]]))
        else:
            result = selector.select(case["query"], tree, include_text=True, retain_raw=retain_raw)
            row.update(result.to_dict())
            # Existing downstream mapping, not re-created document chunks. Keep
            # own section text and Markdown line numbers, never label them pages.
            row["retrieved_content"] = [
                {key: mapping[node_id].get(key) for key in ("node_id", "title", "line_num", "text")}
                for node_id in result.node_list]
            row["diagnostic_pass"] = (not result.node_list if case.get("expect_empty")
                                       else bool(targets[case["id"]] & set(result.node_list)))
        rows.append(row)
    report = {"mode": "live" if live else "dry_run", "threshold": selector.threshold,
              "source_sha256": manifest["source_sha256"], "index_sha256": manifest["index_sha256"],
              "note": "Section-hit diagnostics, not exhaustive relevance labels or a precision/recall benchmark.",
              "cases": rows}
    write_json(output_dir / (f"{case_id}_{'live' if live else 'plan'}.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--document", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--allow-changed-source", action="store_true")
    run = sub.add_parser("evaluate")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--case", default="all")
    run.add_argument("--threshold", type=float)
    run.add_argument("--live", action="store_true", help="Allow paid TypeSafe requests; default is offline.")
    run.add_argument("--retain-raw", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.document, args.output_dir,
                                 allow_changed_source=args.allow_changed_source), indent=2))
    else:
        report = evaluate(args.output_dir, case_id=args.case, threshold=args.threshold,
                          live=args.live, retain_raw=args.retain_raw)
        print(json.dumps({"mode": report["mode"], "cases": len(report["cases"]),
                          "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
