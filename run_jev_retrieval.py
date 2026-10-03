"""Select PageIndex tree nodes with JEV; never runs indexing or answer generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pageindex.jev_retrieval import JevNodeSelector
from pageindex.utils import create_node_mapping


def load_tree(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # CLI accepts on-disk md_to_tree output, get_tree response, or a bare tree.
    # The library itself retains utils' stricter node-tree-only interface.
    if isinstance(data, dict) and "node_id" not in data:
        if "structure" in data:
            return data["structure"]
        if "result" in data:
            return data["result"]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model")
    parser.add_argument("--node-id", action="append", dest="node_ids")
    parser.add_argument("--include-text", action="store_true",
                        help="Send existing node text too (default: title/summary/metadata only).")
    parser.add_argument("--include-content", action="store_true",
                        help="Also save original selected nodes from create_node_mapping.")
    parser.add_argument("--retain-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare requests without an API call/key.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = {key: value for key, value in {
        "jev_batch_size": args.batch_size, "jev_model": args.model,
        "jev_retrieval_threshold": args.threshold,
    }.items() if value is not None}
    selector = JevNodeSelector(config=config)
    tree = load_tree(args.index)
    output = {"schema_version": 1, "query": args.query,
              "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest()}
    if args.dry_run:
        output.update(mode="dry_run", threshold=selector.threshold,
                      requests=selector.prepare_requests(args.query, tree, node_ids=args.node_ids,
                                                         include_text=args.include_text))
    else:
        result = selector.select(args.query, tree, node_ids=args.node_ids,
                                 include_text=args.include_text, retain_raw=args.retain_raw)
        output.update(mode="live", **result.to_dict())
        if args.include_content:
            mapping = create_node_mapping(tree)
            output["retrieved_nodes"] = [mapping[node_id] for node_id in result.node_list]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved {output['mode']} result to {args.output}")


if __name__ == "__main__":
    main()
