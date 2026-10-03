# JEV probability-based node retrieval

## What was replaced, and what was deliberately not replaced

In this checkout the fixed LLM node-selection implementation is **Step 2.1 of
`cookbook/pageindex-vision-rag.ipynb`**, not a function in the PDF indexer.
Previously it removed `text` from the entire tree, inserted the query and tree
into a prompt, and called `call_vlm(search_prompt)`. The generated response was
parsed with `json.loads`, expecting `thinking` and `node_list`. Step 2.2 mapped
those string IDs through `utils.create_node_mapping(..., include_page_ranges=True)`
and `get_page_images_for_nodes`, deduplicating inclusive page ranges in selected
node order. A separate VLM call then answered using the retrieved page images.

The notebook now calls `JevNodeSelector.select` and consumes its `node_list`.
The page-range/image helper and visual answer model are unchanged. An empty
selection returns no images and skips answer generation rather than answering
without document evidence. All notebook outputs were cleared to avoid presenting
old LLM results as JEV results; no live JEV results are included.

The modern SDK's `PageIndexClient.chat()` / `chat_completions()` in `client.py`,
`local_chat.py`, and `agent_tools.py` use a different, tool-calling agent loop.
They do **not** call this fixed node-list selector and are intentionally unchanged.
The deprecated cloud retrieval endpoint is implemented server-side and cannot
be replaced by editing this repository. This change is therefore a replacement
of the checked-in fixed selection example, plus a reusable selector and CLI,
**not a global replacement of every PageIndex chat or cloud search engine**.

Indexing, PDF parsing, tree construction/optimization, Markdown parsing, existing
node IDs, and unrelated LLM functions have not been changed.

## Decision flow and representation

```text
existing tree (all nodes, or caller-supplied node_ids)
  -> independent JEV Noul questions, grouped into requests
  -> ScoredNode(node_id, probability) for every evaluated node
  -> Python probability >= threshold
  -> result.node_list: list[str]
  -> existing node mapping -> original text / page ranges / page images
```

The original search was a **single full-tree selection stage**, not recursive
parent-gated retrieval. Every node is scored independently in depth-first tree
order, so a rejected parent never hides a relevant child. A supplied `node_ids`
subset is evaluated in the caller's order. No IDs are renumbered, no top-k fallback
is applied, and no below-threshold node is substituted when the selection is empty.

```python
from pageindex.jev_retrieval import JevNodeSelector, filter_node_ids
from pageindex.utils import create_node_mapping

# tree is the node list (not the get_tree response envelope).
selector = JevNodeSelector(config={"jev_retrieval_threshold": 0.7})
result = selector.select("What does the document say about digital twins?", tree)
print(result.to_dict())
original_nodes = create_node_mapping(tree)
selected_nodes = [original_nodes[node_id] for node_id in result.node_list]

# Tune a threshold from saved scores without another paid request.
more_ids = filter_node_ids(result.scored_nodes, 0.5)
```

Illustrative output, **not a recorded model run**:

```json
{
  "scored_nodes": [
    {"node_id": "0001", "probability": 0.91},
    {"node_id": "0002", "probability": 0.67}
  ],
  "threshold": 0.7,
  "node_list": ["0001"],
  "provider_trace": []
}
```

Actual nonempty selections also contain per-request model/version, token usage,
and question-to-node mappings in `provider_trace`. `retain_raw=True` adds the raw
provider response there. There is deliberately no fabricated `thinking` field:
JEV supplies a probability, not an LLM-generated rationale.

## Verified TypeSafe contract

References checked 2026-10-03:
[primitives](https://docs.typesafe.ai/primitives),
[Noul](https://docs.typesafe.ai/primitives/noul),
[HTTP API](https://docs.typesafe.ai/api),
[models and limits](https://docs.typesafe.ai/models).

The implementation uses `POST https://api.typesafe.ai/v1/systemone`, bearer
authentication, `model`, shared JSON `state`, and a map of `questions`. Each node
question has `type: "noul"`, explicit instructions and true/false criteria.
`answers[question_id].noul` is **P(relevant = yes)**. It is not the unrelated
Choice/Score `confidence` field, and a low probability means a confident no,
not a relevant result with low confidence.

The API evaluates questions independently in parallel inside a request. Each
question's instructions identify its exact tree path, e.g. `tree[0].nodes[1]`,
because question IDs themselves are not exposed to the model. Matching uses
returned question IDs rather than dictionary/response position.

All batches share the original nested tree, including titles, summaries,
prefix summaries, coordinates and metadata. As in the original notebook,
`text` is omitted by default. `include_text=True` retains the original text
field too; this is useful for Markdown indexes without generated summaries.
The prompt asks about each node's own section, consistent with downstream local
page-range and Markdown-section retrieval. No new chunk representation is built.

## Configuration and boundaries

Use the existing `ConfigLoader`/`pageindex/config.yaml` convention:

| Key | Default | Meaning |
| --- | --- | --- |
| `jev_model` | `jev-latest` | Documented alias; pin a supported version for comparisons. |
| `jev_endpoint` | `https://api.typesafe.ai/v1/systemone` | HTTPS endpoint; no embedded credentials. |
| `jev_retrieval_threshold` | `0.5` | Inclusive selection boundary, in `[0, 1]`. |
| `jev_batch_size` | `32` | Independent questions per request. |
| `jev_timeout` | `30` | Requests connection/read timeout, seconds. |
| `jev_max_retries` | `2` | Additional retryable HTTP attempts per batch. |

Threshold, batch size and timeout/retry defaults are application choices, **not**
provider limits or calibrated retrieval guarantees. They can be overridden by
`JevNodeSelector(config={...})`; `select(..., threshold=...)` overrides the threshold
for one call. At threshold `0.7`, probability `0.7` passes and `0.699999` does not.
Threshold zero admits all valid scores including zero; threshold one only admits
one. Booleans, strings, NaN, infinities and out-of-range values are rejected.
Thresholds never enter the JEV prompt or response parser.

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY`, or pass `api_key=` explicitly.
The existing dotenv setup loads `.env`. No key is required for request preparation,
empty candidate sets, or normal mocked tests. OpenAI/PageIndex keys are only
required for their separate indexing/cloud/answer features, not this selector.

### Limits and error behavior

The documented model limits are 64k tokens for state plus all questions, and 32k
for state plus the longest question. The provider's tokenizer is authoritative;
this integration does not pretend that characters equal tokens. Reducing batch
size helps the first bound but **does not shrink the shared state**. Over-limit
requests fail clearly; there is no automatic text truncation or parent pruning.
Large documents require deliberately selecting an appropriate existing subtree
as input or preparing a smaller existing context. A `node_ids` subset alone does
not remove the surrounding tree from state. JEV receives text/JSON, not page images.

Wrong/missing/extra answers, duplicate JSON keys, duplicate/missing node IDs,
wrong answer types, and malformed probabilities raise `JevRetrievalError`, a
`PageIndexAPIError` subclass. A failed later batch invalidates the entire selection;
partial results are not silently returned. HTTP 429/529 and selected transient
5xx responses use bounded backoff with Retry-After support. A Retry-After above
60 seconds raises instead of sleeping indefinitely. Authentication, validation,
redirect, malformed-response and ambiguous network/timeout failures do not
silently retry or fall back to the old LLM. Automatic retries can incur costs.

Enable `logging.getLogger("pageindex.jev_retrieval").setLevel(logging.DEBUG)` with
a logging handler to see node IDs, returned probabilities and threshold outcomes.
Logs omit credentials and document/query text. Raw responses and CLI request plans
remain opt-in and should still be treated as private. Keep generated indexes,
request plans and live results under ignored `.pageindex/`, not in Git.

## Standalone CLI

From the repository root, after installing requirements:

```bash
python run_jev_retrieval.py \
  --index /path/to/existing_structure.json \
  --query "Explain digital twins" \
  --threshold 0.7 --include-text --include-content \
  --output .pageindex/retrieval.json
```

The CLI accepts a bare tree, `md_to_tree`'s `structure` envelope, or `get_tree`'s
`result` envelope. `--dry-run` saves the requests without an API call. Other
options include `--node-id` (repeatable), `--batch-size`, `--model`, and
`--retain-raw`. `--include-content` uses the existing mapping and returns the
original selected nodes. It does not run indexing or answer generation.

For the vision-RAG notebook, install this **checkout**, not the unmodified PyPI
package. The install cell assumes the notebook is launched from `cookbook/`.
The notebook's existing PageIndex cloud and visual-answer prerequisites still apply.

## Local-agent test: the supplied MGTM03 guide

Source supplied by the user:

```text
ai_sharefolder/mba_course/903_digital_innovation/MGTM03_Digital_Enterprise_and_Innovation_Teacher_Preparation_Guide.md
```

The exact downloaded Markdown is 72,437 bytes; SHA-256:

```text
2138396663a47292a6c40c2951d69925e5f1abcf389e3702417b0d7687c5bcaa
```

**The private document and derived index are not committed to this public repo.**
Obtain the original from the user's Drive or the private prepared fixture archive.
Do not reformat it. `--allow-changed-source` is an explicit override for an updated
source, not a workaround to mask accidental changes. The recorded source and
index checksums make every later run traceable.

```bash
git switch feat/jev-node-retrieval
python -m venv .venv
# macOS/Linux; Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
cp .env.example .env
# Set TYPESAFE_API_KEY in .env locally; do not print or commit it.

# Offline preparation using the actual Markdown indexer:
python -m examples.jev_retrieval.teacher_guide prepare \
  --document "/path/to/MGTM03_Digital_Enterprise_and_Innovation_Teacher_Preparation_Guide.md"

# Offline inspection of all eight cases, no key/cost:
python -m examples.jev_retrieval.teacher_guide evaluate --threshold 0.7

# The explicit --live flag allows paid JEV requests. Start with ONE case:
python -m examples.jev_retrieval.teacher_guide evaluate \
  --case digital_twins --threshold 0.7 --live --retain-raw

# Only after inspecting the first result, run the complete case set:
python -m examples.jev_retrieval.teacher_guide evaluate \
  --case all --threshold 0.7 --live --retain-raw
```

Preparation calls the existing `md_to_tree` with thinning/summaries/document
description disabled and IDs/text enabled. It requires **no OpenAI, JEV or cloud
requests**, and the exact supplied file produces **226 nodes**. Markdown line
numbers stay line numbers, never invented PDF page numbers. Each case scores
these existing nodes with `include_text=True`. At batch size 32 this is eight
requests per case, or 64 for eight cases, before any HTTP retries.

Outputs in `.pageindex/jev_teacher_guide/`:
`index.json`, `manifest.json`, `all_plan.json`, and per-run `*_live.json` containing
all scores, selected IDs, original selected section text and source line numbers.
The live reports also record actual model versions and token usage. Re-running
the same case overwrites its report; copy reports aside for model/threshold studies.

The cases cover digital twins, design thinking versus agile, sustainability,
3S versus 3A, assessment, an Industry 4.0/5.0 query in Traditional Chinese, and two
unrelated queries. Expected section titles resolve against the actual index
before a paid call. They are **section-hit diagnostics**, not exhaustive relevance
labels or a precision/recall benchmark. A diagnostic miss is saved as
`diagnostic_pass: false`; it does not turn the runner into a hard CI quality gate.
Inspect content and false positives manually before choosing a production threshold.

## Automated validation

```bash
python -m pytest -q tests/test_jev_retrieval.py tests/test_jev_teacher_guide.py
python -m pytest -q
```

The focused tests cover the current API shape, shuffled answer order, threshold
boundaries, invalid responses/configuration, multiple batches, parent/child
independence, failures/retries, raw traces, request preparation, real Markdown
indexing, CLI envelopes/content, and execution of the **actual notebook selection
cells followed by the existing page-image helper**. All external JEV calls are
mocked; normal tests never use paid JEV requests or the private guide.

The remaining live validation is the explicit `--live` command above: verify
account authentication, the current provider response contract and limits, then
review real relevance scores and tune the threshold. Mocked success proves the
plumbing, not JEV retrieval quality or end-to-end visual-answer accuracy.
