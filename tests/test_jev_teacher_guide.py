"""The local-agent handoff runs the real Markdown indexer and existing mapping."""
import json
import sys
from unittest.mock import Mock

import pytest
import requests

from examples.jev_retrieval import teacher_guide
from pageindex import utils
from pageindex.jev_retrieval import JevNodeSelector, NodeSelectionResult, ScoredNode
import run_jev_retrieval


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Synthetic/public fixture only. Never put the private teaching guide in CI.
    text = "# Course\n\nOverview.\n\n## 5.8 Digital Twins\n\nA digital representation of an asset.\n"
    source = tmp_path / "guide.md"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "prepared"
    spec = tmp_path / "cases.json"
    spec.write_text(json.dumps({"source": {"sha256": "different"}, "cases": [
        {"id": "twins", "query": "What is a digital twin?", "expected_any_sections": ["5.8 Digital Twins"]},
        {"id": "negative", "query": "Croissant recipe?", "expect_empty": True}]}))
    monkeypatch.setattr(teacher_guide, "SPEC_PATH", spec)
    monkeypatch.setattr(requests.Session, "post", Mock(side_effect=AssertionError("No network")))
    # Also fail if preparation accidentally enables paid summaries.
    monkeypatch.setattr(utils, "llm_completion", Mock(side_effect=AssertionError("No LLM")))
    monkeypatch.setattr(utils, "llm_acompletion", Mock(side_effect=AssertionError("No LLM")))
    manifest = teacher_guide.prepare(source, output, allow_changed_source=True)
    return source, output, manifest


def test_real_markdown_preparation_and_offline_plan(prepared):
    source, output, manifest = prepared
    assert manifest["node_count"] == 2 and manifest["llm_calls"] == 0
    assert manifest["expected_source_match"] is False
    tree = json.loads((output / "index.json").read_text())["structure"]
    mapping = utils.create_node_mapping(tree)
    child = next(n for n in mapping.values() if n["title"] == "5.8 Digital Twins")
    assert "digital representation" in child["text"]
    assert child["line_num"] == 5
    plan = teacher_guide.evaluate(output, threshold=.7)
    assert plan["mode"] == "dry_run"
    assert plan["cases"][0]["expected_target_ids"] == [child["node_id"]]
    assert plan["cases"][1]["expected_target_ids"] == []
    assert plan["cases"][0]["request_count"] == 1


def test_checksum_mismatch_and_changed_index_fail(prepared):
    source, output, _ = prepared
    with pytest.raises(ValueError, match="checksum"):
        teacher_guide.prepare(source, output)
    (output / "index.json").write_text("{}")
    with pytest.raises(ValueError, match="index changed"):
        teacher_guide.evaluate(output, live=True)


def test_missing_expected_sections_validated_before_paid_requests(prepared):
    _, output, _ = prepared
    index_path = output / "index.json"
    data = json.loads(index_path.read_text())
    data["structure"][0]["nodes"][0]["title"] = "Different title"
    index_path.write_text(json.dumps(data))
    import hashlib
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="section titles absent"):
        teacher_guide.evaluate(output, live=True)


def test_mocked_live_flow_reaches_original_markdown_text(prepared, monkeypatch):
    _, output, _ = prepared

    def fake_select(self, query, tree, **kwargs):
        assert kwargs["include_text"] is True
        mapping = utils.create_node_mapping(tree)
        return NodeSelectionResult(tuple(ScoredNode(node_id,
            .7 if "digital twin" in query and node["title"] == "5.8 Digital Twins" else .1)
            for node_id, node in mapping.items()), self.threshold)

    monkeypatch.setattr(JevNodeSelector, "select", fake_select)
    report = teacher_guide.evaluate(output, live=True, threshold=.7)
    assert all(case["diagnostic_pass"] for case in report["cases"])
    content = report["cases"][0]["retrieved_content"]
    assert len(content) == 1 and content[0]["line_num"] == 5
    assert "digital representation" in content[0]["text"]
    assert report["cases"][1]["retrieved_content"] == []


def test_generic_cli_dry_run_and_original_content(prepared, monkeypatch):
    _, output, _ = prepared
    result_path = output / "cli.json"
    args = ["run_jev_retrieval.py", "--index", str(output / "index.json"), "--query", "twins",
            "--threshold", "0.7", "--include-text", "--output", str(result_path)]
    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    run_jev_retrieval.main()
    plan = json.loads(result_path.read_text())
    assert plan["mode"] == "dry_run" and plan["threshold"] == .7
    assert len(plan["requests"][0]["question_node_ids"]) == 2
    assert "text" in plan["requests"][0]["payload"]["state"]["tree"][0]
    monkeypatch.setattr(JevNodeSelector, "select", lambda *a, **k:
                        NodeSelectionResult((ScoredNode("0000", .7), ScoredNode("0001", .1)), .7))
    monkeypatch.setattr(sys, "argv", args + ["--include-content"])
    run_jev_retrieval.main()
    result = json.loads(result_path.read_text())
    assert result["node_list"] == ["0000"]
    assert "Overview." in result["retrieved_nodes"][0]["text"]
    assert "provider_trace" in result


def test_cli_loads_existing_envelopes_and_bare_tree(tmp_path):
    tree = [{"node_id": "0001", "title": "T"}]
    path = tmp_path / "tree.json"
    for content in (tree, {"structure": tree}, {"status": "completed", "result": tree}):
        path.write_text(json.dumps(content))
        assert run_jev_retrieval.load_tree(path) == tree
