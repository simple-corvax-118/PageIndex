"""JEV wire contract, threshold semantics and the real notebook handoff; no API calls."""
import ast
import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from pageindex import utils
from pageindex.jev_retrieval import (
    JevNodeSelector, JevRetrievalError, NodeSelectionResult,
    ScoredNode, filter_node_ids, parse_jev_response,
)

ROOT = Path(__file__).resolve().parents[1]


def wire(values):
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 100, "output_tokens": 3},
            "answers": {key: {"type": "noul", "noul": value} for key, value in values.items()}}


def response(data=None, status=200, text=None, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = (text if text is not None else json.dumps(data)).encode()
    r.headers.update(headers or {})
    r.close = Mock()
    return r


@pytest.fixture
def tree():
    return [{"node_id": "0000", "title": "Parent", "prefix_summary": "Parent introduction",
             "start_index": 1, "end_index": 1, "text": "parent text",
             "nodes": [{"node_id": "0001", "title": "Relevant child", "summary": "Details",
                        "start_index": 2, "end_index": 3, "text": "child evidence"}]},
            {"node_id": "0002", "title": "Sibling", "start_index": 3, "end_index": 4,
             "metadata": {"source": "original"}, "text": "sibling evidence"}]


@pytest.fixture(autouse=True)
def block_external(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(requests.Session, "post", Mock(side_effect=AssertionError("No live API in tests")))


def test_parse_maps_by_question_id_not_answer_order():
    scores = parse_jev_response(wire({"second": .67, "first": .91}),
                                {"first": "0007", "second": "0011"})
    assert scores == (ScoredNode("0007", .91), ScoredNode("0011", .67))
    assert NodeSelectionResult(scores, .7).to_dict() == {
        "scored_nodes": [{"node_id": "0007", "probability": .91},
                         {"node_id": "0011", "probability": .67}],
        "threshold": .7, "node_list": ["0007"], "provider_trace": []}


@pytest.mark.parametrize("threshold,expected", [
    (0, ["a", "b", "c", "d"]), (.7, ["c", "d"]), (1, ["d"])])
def test_inclusive_threshold_and_no_ranking_reorder(threshold, expected):
    scores = tuple(ScoredNode(n, p) for n, p in zip("abcd", [0, .699999, .7, 1]))
    assert filter_node_ids(scores, threshold) == expected


@pytest.mark.parametrize("invalid", [None, "0.9", True, False, -.1, 1.1, float("nan"),
                                       float("inf"), -float("inf"), 10**500])
def test_invalid_probabilities_and_thresholds(invalid):
    with pytest.raises(JevRetrievalError, match="finite number"):
        parse_jev_response(wire({"q": invalid}), {"q": "0001"})
    with pytest.raises(JevRetrievalError, match="finite number"):
        filter_node_ids([], invalid)


@pytest.mark.parametrize("change", [
    lambda r: r.pop("model"), lambda r: r.update(model=""),
    lambda r: r.pop("usage"), lambda r: r["usage"].update(input_tokens=True),
    lambda r: r["usage"].update(output_tokens=-1),
    lambda r: r["answers"].clear(),
    lambda r: r["answers"].update(other={"type": "noul", "noul": .9}),
    lambda r: r["answers"]["q"].update(type="choice", confidence=.99),
    lambda r: r["answers"]["q"].pop("noul"),
    lambda r: r["answers"].update(q=None),
])
def test_invalid_or_incomplete_contract(change):
    data = wire({"q": .9})
    change(data)
    with pytest.raises(JevRetrievalError):
        parse_jev_response(data, {"q": "0001"})


def test_non_object_response_and_duplicate_mapping():
    with pytest.raises(JevRetrievalError):
        parse_jev_response([], {"q": "0001"})
    with pytest.raises(JevRetrievalError, match="duplicate"):
        parse_jev_response(wire({"x": .9, "y": .8}), {"x": "a", "y": "a"})
    with pytest.raises(JevRetrievalError, match="Duplicate"):
        filter_node_ids([ScoredNode("a", .9), ScoredNode("a", .8)], .7)


def test_prepare_preserves_context_ids_and_input_without_text(tree):
    before = copy.deepcopy(tree)
    selector = JevNodeSelector(config={"jev_batch_size": 2})
    batches = selector.prepare_requests("query", tree)
    assert tree == before
    assert [b["question_node_ids"] for b in batches] == [
        {"node_0": "0000", "node_1": "0001"}, {"node_2": "0002"}]
    assert batches[0]["payload"]["state"] == batches[1]["payload"]["state"]
    state = batches[0]["payload"]["state"]
    assert state["tree"][0]["prefix_summary"] == "Parent introduction"
    assert state["tree"][1]["metadata"] == before[1]["metadata"]
    assert '\"text\"' not in json.dumps(state)
    questions = batches[0]["payload"]["questions"]
    assert "`tree[0].nodes[0]`" in questions["node_1"]["instructions"]
    assert 'threshold' not in json.dumps(batches)
    assert all(q["type"] == "noul" for q in questions.values())
    with_text = selector.prepare_requests("q", tree, include_text=True)[0]
    assert with_text["payload"]["state"]["tree"] == before


def test_subset_and_single_root(tree):
    selector = JevNodeSelector()
    batch = selector.prepare_requests("q", tree, node_ids=["0002", "0001"])[0]
    assert list(batch["question_node_ids"].values()) == ["0002", "0001"]
    single = selector.prepare_requests("q", tree[0])[0]
    assert "`tree.nodes[0]`" in single["payload"]["questions"]["node_1"]["instructions"]
    for ids in (["unknown"], ["0001", "0001"], "0001", [1]):
        with pytest.raises(JevRetrievalError):
            selector.prepare_requests("q", tree, node_ids=ids)


@pytest.mark.parametrize("bad_tree", [
    [{"title": "missing id"}], [{"node_id": 1}], [{"node_id": ""}],
    [{"node_id": "a"}, {"node_id": "a"}], [{"node_id": "a", "nodes": {}}],
    [{"node_id": "a", "nodes": [None]}], [{"node_id": "a", "metadata": float("nan")}],
])
def test_invalid_tree_fails_before_network(bad_tree):
    with pytest.raises(JevRetrievalError):
        JevNodeSelector().prepare_requests("q", bad_tree)


def test_cycle_and_invalid_query(tree):
    tree[0]["nodes"].append(tree[0])
    with pytest.raises(JevRetrievalError, match="cyclic"):
        JevNodeSelector().prepare_requests("q", tree)
    for query in (None, "", " "):
        with pytest.raises(JevRetrievalError, match="query"):
            JevNodeSelector().prepare_requests(query, [])


def test_empty_is_not_a_missing_key_failure(tree):
    assert JevNodeSelector().select("q", []).node_list == []
    assert JevNodeSelector().select("q", tree, node_ids=[]).scored_nodes == ()
    with pytest.raises(JevRetrievalError, match="TYPESAFE_API_KEY"):
        JevNodeSelector().select("q", tree)


def test_batches_rejected_parent_selected_child_and_trace(tree, caplog):
    session = Mock()
    session.post.side_effect = [response(wire({"node_1": .7, "node_0": .1})),
                                response(wire({"node_2": .6999}))]
    selector = JevNodeSelector("test-secret", config={"jev_batch_size": 2}, session=session)
    with caplog.at_level("DEBUG", logger="pageindex.jev_retrieval"):
        result = selector.select("q", tree, threshold=.7, retain_raw=True)
    assert result.node_list == ["0001"]  # Rejected root did not prune this child.
    assert [s.probability for s in result.scored_nodes] == [.1, .7, .6999]
    assert len(result.provider_trace) == 2
    assert result.provider_trace[0]["model"] == "jev-1.13.0"
    assert result.provider_trace[0]["raw_response"]["answers"]["node_1"]["noul"] == .7
    assert "0001" in caplog.text and "passed=True" in caplog.text
    assert "test-secret" not in caplog.text
    session.close.assert_not_called()
    assert session.post.call_args.kwargs["timeout"] == 30
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert session.post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-secret"
    # Re-threshold saved scores without another provider request.
    assert filter_node_ids(result.scored_nodes, .05) == ["0000", "0001", "0002"]
    assert session.post.call_count == 2


def test_owned_session_closes_default_trace_and_environment_key(tree, monkeypatch):
    session = Mock()
    session.post.return_value = response(wire({"node_0": .1, "node_1": .8, "node_2": .2}))
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-secret")
    result = JevNodeSelector().select("q", tree)
    assert "raw_response" not in result.provider_trace[0]
    session.close.assert_called_once()
    assert session.post.call_args.kwargs["headers"]["Authorization"] == "Bearer env-secret"


def test_incomplete_later_batch_aborts_entire_selection(tree):
    session = Mock()
    session.post.side_effect = [response(wire({"node_0": .9, "node_1": .9})), response(wire({}))]
    with pytest.raises(JevRetrievalError, match="batch 1.*0002"):
        JevNodeSelector("test", config={"jev_batch_size": 2}, session=session).select("q", tree)


@pytest.mark.parametrize("status", [301, 401, 422, 500, 529])
def test_http_failures_are_not_negative_decisions(tree, status):
    session = Mock()
    session.post.return_value = response({}, status=status)
    with pytest.raises(JevRetrievalError, match=f"HTTP {status}"):
        JevNodeSelector("test", config={"jev_max_retries": 0}, session=session).select("q", tree)
    assert session.post.call_count == 1


@pytest.mark.parametrize("body", ["not JSON", '{"model":"one","model":"two"}'])
def test_invalid_json_and_duplicate_keys_fail(tree, body):
    session = Mock()
    session.post.return_value = response(text=body)
    with pytest.raises(JevRetrievalError, match="invalid JSON"):
        JevNodeSelector("test", session=session).select("q", tree)


def test_timeout_is_sanitized_and_not_retried(tree):
    session = Mock()
    session.post.side_effect = requests.Timeout("contains secret")
    with pytest.raises(JevRetrievalError, match="network/timeout") as error:
        JevNodeSelector("test", session=session).select("q", tree)
    assert "contains secret" not in str(error.value)
    assert session.post.call_count == 1


def test_retry_429_529_with_backoff(tree, monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("pageindex.jev_retrieval.time.sleep", sleep)
    session = Mock()
    responses = [response({}, 429, headers={"Retry-After": "3"}), response({}, 529),
                 response(wire({"node_0": .9, "node_1": .8, "node_2": .1}))]
    session.post.side_effect = responses
    assert JevNodeSelector("test", session=session).select("q", tree).node_list == ["0000", "0001"]
    assert [call.args[0] for call in sleep.call_args_list] == [3, 2]
    for r in responses:
        r.close.assert_called_once()


@pytest.mark.parametrize("config", [
    {"jev_batch_size": 0}, {"jev_batch_size": True}, {"jev_max_retries": -1},
    {"jev_timeout": 0}, {"jev_timeout": float("inf")}, {"jev_model": ""},
    {"jev_retrieval_threshold": 1.1}, {"jev_endpoint": "http://example.test"},
    {"jev_endpoint": "https://user:secret@example.test"},
    {"jev_endpoint": "https://example.test?api_key=secret"},
    {"jev_endpoint": "https://[invalid"},
])
def test_invalid_configuration(config):
    with pytest.raises(JevRetrievalError):
        JevNodeSelector(config=config)


@pytest.mark.parametrize("probabilities,expected_pages", [([.1, .8, .9], [2, 3, 4]),
                                                         ([.1, .2, .3], [])])
def test_actual_notebook_selection_to_existing_page_image_helper(tree, probabilities,
                                                                expected_pages, monkeypatch):
    notebook = json.loads((ROOT / "cookbook/pageindex-vision-rag.ipynb").read_text())
    session = Mock()
    session.post.return_value = response(wire(dict(zip(["node_0", "node_1", "node_2"], probabilities))))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-notebook")
    monkeypatch.setattr(requests, "Session", lambda: session)
    vlm = Mock(side_effect=AssertionError("LLM must not select nodes"))
    ns = {"tree": tree, "utils": utils, "total_pages": 4,
          "page_images": {n: f"page-{n}.jpg" for n in range(1, 5)}, "call_vlm": vlm}
    helper_ast = ast.parse("".join(notebook["cells"][13]["source"]))
    helper = next(n for n in helper_ast.body if isinstance(n, ast.FunctionDef)
                  and n.name == "get_page_images_for_nodes")
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "notebook-helper", "exec"), ns)
    code = compile("".join(notebook["cells"][21]["source"]), "notebook-selection", "exec",
                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    asyncio.run(eval(code, ns))
    for index in (23, 25):
        exec("".join(notebook["cells"][index]["source"]), ns)
    assert ns["retrieved_page_images"] == [f"page-{n}.jpg" for n in expected_pages]
    assert ns["retrieved_nodes"] == ns["selection"].node_list
    assert ns["node_map"]["0001"]["node"] is tree[0]["nodes"][0]
    assert len(ns["tree_search_result_json"]["scored_nodes"]) == 3
    vlm.assert_not_called()
    if not expected_pages:
        answer_code = compile("".join(notebook["cells"][28]["source"]), "notebook-answer", "exec",
                              flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        asyncio.run(eval(answer_code, ns))
        vlm.assert_not_called()
        assert "No nodes passed" in ns["answer"]
