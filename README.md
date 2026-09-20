# SNAIL — Single Node Activated Inference Layer

[![PyPI version](https://img.shields.io/pypi/v/snail-dsl.svg)](https://pypi.org/project/snail-dsl/)
[![Python versions](https://img.shields.io/pypi/pyversions/snail-dsl.svg)](https://pypi.org/project/snail-dsl/#files)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lordxmen2k/SNAIL-DSL/blob/main/LICENSE)
[![Status: Beta](https://img.shields.io/pypi/status/snail-dsl.svg)](https://pypi.org/project/snail-dsl/)
[![Downloads](https://img.shields.io/pypi/dm/snail-dsl.svg)](https://pypistats.org/packages/snail-dsl)

**A Python DSL for composing frozen, single-pass neural primitives into statically-typed dataflow programs. Deterministic by design — the author composes the workflow as a typed DAG, the model fills in the leaves.**

SNAIL is the alternative to agentic LLM frameworks. Author composes, model fills leaves, OOD is a first-class return type, every run emits a tamper-evident manifest. There is no loop the model can get stuck in, because the execution graph has no back-edge worth calling one. Compared to LangChain / Claude Agent SDK / AutoGPT, SNAIL is structurally cheaper to bound, structurally easier to test, and structurally audit-ready.

```bash
pip install snail-dsl
```

**You do not need an API key to start.** SNAIL ships with a stub provider so you can build, run, and test programs without any external service.

---

## 📖 Glossary — every word SNAIL uses (read this first)

Before you install anything or write any code, read this section. Every term you'll see in the rest of the README is defined here in plain English. If a word below is unfamiliar, the rest of the README won't make sense — that's why this comes first.

### Concepts

| Word | What it means in SNAIL (plain English) |
|---|---|
| **DSL** | "Domain-Specific Language." SNAIL is a small Python-based language for one specific purpose: building AI workflows out of small pieces. It looks like Python but adds extra rules and helpers. |
| **Recipe** | The instructions for building an AI workflow. In code, it's a Python file with `@node` decorators and a `Program(...)` call. In training, it's a YAML file with `name`, `dataset`, `nodes`, and `golden_cases`. **A recipe describes what should happen. A program is the actual thing that runs.** |
| **Node** | One step in a recipe. It's a Python function wrapped with `@node`. When the program runs, each node fires exactly once and produces one output. Think of a recipe with steps — each step is a node. |
| **Program** | The whole recipe wired up. It's a `Program(...)` object that holds a list of nodes and a list of edges (the connections between nodes). You call `program.run(input)` to execute it. |
| **DAG** | "Directed Acyclic Graph." A flow chart with no loops. Node A feeds Node B feeds Node C. The arrows (edges) point one way and there are no cycles. **Every SNAIL program is a DAG** — that's the entire structure. |
| **Edge** | A connection between two nodes, telling SNAIL "the output of this node flows into that node." Built with `edge(node.ok)` or `edge(node.ood)`. |
| **Escalate** | The `escalate(node, to=other)` helper. A one-liner that wires the OOD branch of `node` into `other`. Used for the small-model-first / large-model-on-OOD pattern. Rejects downgrades (you can't escalate to a cheaper tier) and cycles at composition time. |
| **Parallel edges** | The `parallel_edges(inputs=node, to=[list])` helper. Multiple nodes receive the same input, each as a parallel branch. Their results feed downstream as separate typed fields. |
| **Cost tier** | A label on each node — "small", "medium", or "large". Used by `escalate()` to forbid cheapening. Default is "medium" for `HostedNode` and "small" for `DeterministicNode`. |
| **Weight pin** | A string like `phi-4-mini-3.8b@sha256:abc123...` that locks a node to a specific SHA-256 of its weights file. `Program(...)` raises `WeightPinMismatch` at construction time if the file's hash doesn't match. The "frozen and hash-verified" discipline, made literal. |
| **ECE** | "Expected Calibration Error." A number between 0 and 1 that measures how well a model's reported confidence matches its actual accuracy. `snail calibrate` runs the program's golden cases and emits ECE; a low ECE means the confidence threshold isn't wishful. |
| **Frozen weights** | A saved snapshot of a model's parameters (the numbers it learned during training). Once frozen, they never change. Loading a model gives you frozen weights. The word "frozen" means: locked, immutable, can't be silently upgraded. |
| **Distribution** | The "training data" a node was trained on. If new input looks very different from that data, the node says "I don't know" instead of guessing wrong. Example: `distribution="customer_intents_v3"` means "this node was trained on the v3 customer-intents dataset." |
| **OOD** | "Out-Of-Distribution." When input doesn't match what the node was trained on. SNAIL doesn't crash on OOD — it returns a special `OODSignal` saying "I'm not confident, please handle this case explicitly." |
| **OK / OOD result** | Every node's output has TWO possible shapes: `ok` (with the answer) or `ood` (saying "I don't know"). The schema is `class MyResult(NodeResult): ok: ... | None = None; ood: OODSignal | None = None`. The program decides what to do with each. |
| **Confidence threshold** | A number between 0 and 1. If the node's confidence is below this number, the result is **automatically flipped to OOD**. Default is 0.5. Set it to 0.7 to be more strict. |
| **Confidence score** | A number between 0 and 1 that the model itself reports: "I'm 0.95 confident in this answer." Comes from the node's output. If it's below the threshold, SNAIL flips to OOD. |
| **Manifest** | A structured log of what happened during one `run()` call. It records which nodes fired, in what order, how long each took, whether each returned `ok` or `ood`, the tokens in/out and cost per node, and any errors. You can read it after `run()` to audit the program's behavior. |
| **Provider** | A way to talk to a language model. SNAIL has built-in providers: `stub` (no network, default), `anthropic` (Claude), `openai` (GPT and OpenAI-compatible APIs), `ollama` (local models). You pick the provider via `provider="anthropic"` etc. |
| **Wrapper** | A factory that turns an external thing (a hosted API, a local model, a pure function) into a SNAIL node. The three wrappers are `HostedNode`, `ExternalLocalNode`, `DeterministicNode`. |
| **CLI** | "Command-Line Interface." The `snail` command you type in your terminal. It has five subcommands: `run`, `inspect`, `render`, `train`, `verify`. |

### Files and outputs

| Term | What it is |
|---|---|
| **Frozen weights file** | A `.snail.json` file produced by training a recipe. Contains the recipe spec, the training settings, and a sha256 hash so you can verify the file wasn't tampered with. Format: `snail-json-v1`. |
| **Trace file** | A `<node>_trace.json` file produced during training. Contains the input/output of each sample the model saw. Useful for debugging what the training actually did. |
| **Golden case** | A tiny test stored as a JSON file: "if the input is X, the output should be Y." The verifier runs all golden cases against your program and reports which ones fail. |
| **Golden directory** | The folder where you store golden cases. Layout: `goldens/golden/<node-name>/<case-id>.json`. |
| **SVG** | "Scalable Vector Graphics." An image format that's just XML. SNAIL renders your program as an SVG file so you can open it in a browser and see the nodes + edges visually. |
| **YAML** | "YAML Ain't Markup Language." A text format for config files. Recipes are written in YAML. Looks like `key: value` with indentation. |

### Commands in the CLI

| Command | What it does |
|---|---|
| `snail run <file> --input '<json>'` | Run a program once and print its manifest. |
| `snail inspect <file>` | Print a program's structure: nodes, edges, distributions. No execution. |
| `snail render <file> --out <svg>` | Render the DAG to an SVG file you can open in a browser. |
| `snail train <recipe.yaml> --output-dir <dir>` | Train a recipe: emit a frozen weights file + a trace file. |
| `snail verify <file> --golden-dir <dir>` | Run golden cases against the program, exit 0 if all pass, 1 if any fail. |

### Symbols and notation

| Symbol | What it means |
|---|---|
| `@node(...)` | The decorator that wraps a function as a SNAIL node. Anything inside the parentheses is the node's configuration. |
| `node.ok` | A reference to the OK variant of a node's output. Used with `edge()` to say "if this node returns OK, send the result to...". |
| `node.ood` | A reference to the OOD variant. Used with `edge()` to say "if this node returns OOD, send it down the fallback path." |
| `result.is_ok` | True if the result is the OK variant. |
| `result.is_ood` | True if the result is the OOD variant. |
| `result.terminal` | The last node that fired. |
| `result.outputs` | A dictionary: `{node_name: NodeResult}`. Read outputs by name. |
| `result.manifest` | The structured log of the run. |

---

Now that the words are clear, let's install and run.

---

## 📋 Step 0 — Before you start

You will need:

- **A computer** (Windows, macOS, or Linux).
- **Python 3.10, 3.11, or 3.12** installed. To check, open a terminal and type:

  ```bash
  python --version
  ```

  If you see `Python 3.10.x`, `Python 3.11.x`, or `Python 3.12.x`, you are good. If not, [install Python](https://www.python.org/downloads/) and re-run the check.

- **A folder to work in.** Create one called `snail-playground` anywhere you like:

  ```bash
  mkdir snail-playground
  cd snail-playground
  ```

You will **not** need:
- An API key for any LLM (we use the built-in stub, no network).
- A GPU.
- A database.
- An internet connection after the initial `pip install`.

When you have done the above, move to Step 1.

---

## 📦 Step 1 — Install SNAIL

Run this in your terminal:

```bash
pip install snail-dsl
```

Wait for it to finish. You should see lines like:

```
Successfully installed annotated-types-0.8.0 click-8.5.0 httpx-0.28.1 pydantic-2.13.5 pyyaml-6.0.3 snail-dsl-0.2.1 ...
```

Now confirm the install worked:

```bash
python -c "import snail; print(snail.__version__)"
```

Expected output:

```
0.2.1
```

If you see `0.2.1`, the install worked. Move to Step 2.

---

## ✅ Step 2 — Run your first SNAIL program

We are going to type a small program into a file, then run it. This program classifies the intent of a customer message (e.g., "I want a refund" → "refund"). It has two nodes connected together.

### 2.1 — Create the program file

Open your text editor (Notepad on Windows, TextEdit on macOS, VS Code, anything). Create a new file called `hello_snail.py` inside your `snail-playground` folder.

Copy-paste this **exactly** into the file:

```python
"""hello_snail.py — your first SNAIL program.

This program has two nodes connected together:
  1. classify_intent — looks at a customer message and decides if it's
     about a refund, billing, or something else.
  2. extract_entities — based on the intent, pretends to pull out an
     order ID (in real use this would call a real model).

Each node returns either an "ok" result (with the answer) or an "ood"
result (saying "I don't know — this isn't what I was trained on").
"""

# 1. We need these types from SNAIL and pydantic.
#    pydantic is a library that lets us define data shapes in Python.
from pydantic import BaseModel
from snail import (
    node,            # the @node decorator
    Program,         # the DAG container
    edge,            # the connection builder between nodes
    NodeContext,     # per-call info passed to each node body
    NodeResult,      # base class for "ok or ood" result types
    OODSignal,       # the "I don't know" payload
)

# 2. Tell SNAIL what a "good" output looks like for the first node.
#    A node returns either `ok` (success) or `ood` (not sure).
class IntentOk(BaseModel):
    intent: str
    confidence: float  # how sure the model is, 0.0 to 1.0

class Intent(NodeResult):
    ok: IntentOk | None = None     # filled in on success
    ood: OODSignal | None = None   # filled in when OOD

# 3. Wrap a function with @node. This becomes a "node" in the DAG.
@node(
    name="classify_intent",
    input_schema=dict,                  # we'll accept a plain dict
    output_schema=Intent,               # the result must match Intent
    distribution="customer_intents_v3", # the training distribution
    confidence_threshold=0.7,           # below this → OOD
)
def classify_intent(ctx: NodeContext, weights, message: dict):
    text = message.get("text", "").lower()
    if "refund" in text:
        intent = "refund"
    elif "balance" in text:
        intent = "billing"
    else:
        intent = "other"
    # Return "ok" with our answer. SNAIL wraps this in Intent(...).
    return Intent(ok=IntentOk(intent=intent, confidence=0.95))

# 4. A second node that depends on the first.
class EntitiesOk(BaseModel):
    order_id: str | None
    confidence: float

class Entities(NodeResult):
    ok: EntitiesOk | None = None
    ood: OODSignal | None = None

@node(
    name="extract_entities",
    input_schema=Intent,                  # takes the previous node's output
    output_schema=Entities,
    distribution="customer_entities_v3",
    confidence_threshold=0.6,
)
def extract_entities(ctx: NodeContext, weights, intent_result: Intent):
    # If the upstream node was OOD, propagate OOD.
    if intent_result.is_ood:
        return Entities(ood=intent_result.ood)
    # Otherwise extract a fake order ID (in real use, call a real model).
    return Entities(ok=EntitiesOk(order_id="ORD-12345", confidence=0.85))

# 5. Wire them into a Program (the typed DAG).
program = Program(
    name="hello_snail",
    nodes=[classify_intent, extract_entities],
    edges=[edge(classify_intent.ok)],   # classify's OK → extract's input
)

# 6. Run it.
if __name__ == "__main__":
    result = program.run({"text": "I want a refund for order #12345"})
    print("Intent:  ", result.outputs["classify_intent"].ok.intent)
    print("Order ID:", result.outputs["extract_entities"].ok.order_id)
```

Save the file.

### 2.2 — Run the program

In the terminal, in your `snail-playground` folder:

```bash
python hello_snail.py
```

Expected output:

```
Intent:   refund
Order ID: ORD-12345
```

If you see those two lines, you just ran your first SNAIL program. 🎉

**What just happened, in plain English:**

1. SNAIL loaded your `classify_intent` node and `extract_entities` node.
2. SNAIL validated that the DAG is well-formed (one node feeds the other, types match).
3. You called `program.run(...)` with a message.
4. SNAIL called `classify_intent` first. It returned `ok` with `intent="refund"`.
5. SNAIL routed that result to `extract_entities` as input.
6. `extract_entities` returned `ok` with `order_id="ORD-12345"`.
7. SNAIL gave you back both outputs in `result.outputs`.

If you see an error, paste the full error message to me (or open an issue on GitHub) and I'll help fix it.

---

## 🔍 Step 3 — Inspect the program with the CLI

SNAIL ships with a command-line tool called `snail`. One of its subcommands tells you what a program contains without running it.

In the terminal:

```bash
snail inspect hello_snail.py
```

Expected output:

```
Program: hello_snail
  Nodes (2):
    - classify_intent (distribution='customer_intents_v3')
    - extract_entities (distribution='customer_entities_v3')
  Edges (1):
    - classify_intent.ok → extract_entities.input
```

This tells you:
- The program has two nodes.
- One edge connects them: the **OK** output of `classify_intent` feeds into the input of `extract_entities`.

"Distribution" is the name of the training data each node was built from (see the glossary at the top of this README).

---

## 🎨 Step 4 — Render the DAG to SVG (draw it as a picture)

Another subcommand draws your program as a picture you can open in any browser:

```bash
snail render hello_snail.py --out hello_snail.svg
```

Open `hello_snail.svg` in any web browser. You'll see a picture with two boxes (your nodes) and a line between them (the edge). The boxes have rounded corners and the background is dark navy — that is the SNAIL visual style.

---

## 🚂 Step 5 — Train a recipe (no real model needed)

Now we teach SNAIL how to "train" a node from a recipe. A **recipe** is a YAML file that says "here is the training data, here are the training settings, here is where to put the frozen weights when done."

In v0.2.1, the trainer is a discipline scaffold: it runs the loop, records what happened, and saves the metadata. The actual model training (calling PyTorch / your framework) is the `model_forward` callback — for now we use a stub that returns canned responses. v0.3.0+ will ship built-in trainers.

### 5.1 — Create the recipe

In your text editor, create a new file called `my_recipe.recipe.yaml` in `snail-playground`. Copy-paste this exactly:

```yaml
name: my_first_recipe
dataset: customer_intents_v3
frozen_output: weights/classify_intent.snail.json
nodes:
  - name: classify_intent
    input_schema: Message
    output_schema: Intent
    distribution: customer_intents_v3
    epochs: 5
    learning_rate: 0.001
    weight_pin: phi-4-mini-3.8b@sha256:placeholder
golden_cases:
  - input: {text: "I want a refund"}
    expected: {intent: refund, confidence_min: 0.7}
  - input: {text: "What is my balance?"}
    expected: {intent: billing, confidence_min: 0.7}
```

What each line means in plain English:
- `name`: the name of this recipe (you can call it anything).
- `dataset`: the training data this recipe trains against.
- `frozen_output`: where to write the frozen weights file (the file that records what was trained).
- `nodes`: the nodes in this recipe (just one here).
  - `name`: the node's name (must match the node in your program).
  - `input_schema` / `output_schema`: the input/output types (described elsewhere in your code).
  - `distribution`: which training distribution the node belongs to.
  - `epochs`: how many passes through the training data (5 here).
  - `learning_rate`: how aggressively the model updates (0.001 = slow).
  - `weight_pin`: a fake hash pinning the model to a specific version.
- `golden_cases`: the test cases to run after training.

Save the file.

### 5.2 — Run the trainer

In the terminal:

```bash
snail train my_recipe.recipe.yaml --output-dir ./my_weights
```

Expected output:

```
  classify_intent → ./my_weights/classify_intent.snail.json
```

You should now have:

```
snail-playground/
├── hello_snail.py
├── my_recipe.recipe.yaml
└── my_weights/
    ├── classify_intent.snail.json   ← the frozen weights file
    └── classify_intent_trace.json   ← per-sample training trace
```

### 5.3 — Look at the frozen file

```bash
cat my_weights/classify_intent.snail.json
```

You'll see something like:

```json
{
  "format": "snail-json-v1",
  "recipe": "my_first_recipe",
  "node": "classify_intent",
  "distribution": "customer_intents_v3",
  "weight_pin": "phi-4-mini-3.8b@sha256:placeholder",
  "epochs": 5,
  "learning_rate": 0.001,
  "samples_seen": 3,
  "final_loss": 0.1,
  "frozen_at": 1758...,
  "model_state": {"stub": true}
}
```

This is the **frozen weights file**. It records exactly what training happened:
- Which recipe produced it.
- Which node it applies to.
- The training settings (epochs, learning rate).
- The weight pin (which exact model version was used).
- The sha256 hash, so anyone can verify the file wasn't tampered with.

You can audit it, ship it, pin a specific version, or reject any program that uses a different version.

---

## 🧪 Step 6 — Verify with golden test cases

Golden cases are tiny tests that say "if the input is X, the output should be Y." SNAIL runs them and tells you if your program matches.

### 6.1 — Create a golden case

Make a folder:

```bash
mkdir -p goldens/golden/classify_intent
```

In your text editor, create `goldens/golden/classify_intent/case.json` with this exact content:

```json
{
  "input": {"text": "I want a refund"},
  "expected": {"intent": "refund", "confidence_min": 0.7}
}
```

Save.

What each line means in plain English:
- `input`: what we send to the program.
- `expected`: what we expect back. `intent: refund` means the field must equal `"refund"`. `confidence_min: 0.7` means the `confidence` field must be at least `0.7` (you can also use `_max` for upper bounds).

### 6.2 — Run the verifier

```bash
snail verify hello_snail.py --golden-dir ./goldens
```

Expected output:

```
0 failures — all golden cases passed.
```

If the verifier exits with code `0`, all golden cases passed. If it exits with code `1`, the verifier prints what failed. (You can check exit codes with `echo $?` on macOS/Linux or `echo %ERRORLEVEL%` on Windows.)

### 6.3 — Try breaking it

Open `goldens/golden/classify_intent/case.json` and change `refund` to `balance`. Save. Run the verifier again:

```bash
snail verify hello_snail.py --golden-dir ./goldens
```

Expected output:

```
1 failure(s):
  - case.json: payload mismatch
```

That's how the verifier catches regressions. Change it back to `refund` to make it pass again.

---

## 🌐 Step 7 — Use a real LLM (only when you want)

Everything up to Step 6 used the **stub provider** — no network calls, no API keys, no cost. When you are ready to wire a real model, set one environment variable and change one word.

### 7.1 — Set the API key

For Anthropic (Claude):

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (Git Bash)
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

For OpenAI (GPT):

```bash
export OPENAI_API_KEY="sk-..."
```

For Ollama (local, no key needed):

```bash
# Install Ollama from https://ollama.com, then pull a model
ollama pull llama4-scout
```

### 7.2 — Switch the provider in your code

In `hello_snail.py`, add a new node that uses a hosted model. Replace `provider="stub"` with one of:

```python
provider="anthropic"   # uses ANTHROPIC_API_KEY env var
provider="openai"      # uses OPENAI_API_KEY env var
provider="ollama"      # uses local Ollama on http://localhost:11434
```

Example:

```python
summarize = HostedNode(
    name="summarize",
    input_schema=dict,
    output_schema=SummaryResult,
    distribution="summarizer_v1",
    endpoint="anthropic://claude-sonnet-5",
    prompt_template="Summarize: {text}",
    api_key_env="ANTHROPIC_API_KEY",
    provider="anthropic",   # ← change this from "stub"
)
```

If the API key is missing, the node returns **OOD** instead of crashing. The discipline is preserved.

---

## 🎉 You are done

You have now used every piece of SNAIL v0.2.1:

- ✅ `@node`, `Program`, `edge()`, `Manifest` (the four primitives)
- ✅ `HostedNode` with stub (default) + Anthropic / OpenAI / Ollama (real)
- ✅ The `snail` CLI (`inspect`, `render`, `run`, `train`, `verify`)
- ✅ Recipe training (`.recipe.yaml` → frozen weights file)
- ✅ Golden test cases (write → verify)

For the philosophy, theory, and architecture behind all this, read *The Frozen Mind* (DOI `10.5281/zenodo.22839893`).

---

## 📚 Reference (skim later, not needed now)

### The four primitives

| Primitive | What it does |
|---|---|
| **`@node`** | Decorator that wraps a function as a frozen, single-pass, OOD-aware node. |
| **`Program`** | Container that builds a typed DAG of nodes. Validates at construction time. |
| **`edge()`** | Builder for typed field-to-field connections between nodes. |
| **`Manifest`** | Structured per-run log: which nodes fired, latency, variant (OK/OOD), errors. |

### The three wrappers

| Wrapper | When to use |
|---|---|
| **`HostedNode`** | Wrap an LLM API call (Anthropic, OpenAI, Ollama, etc.). |
| **`ExternalLocalNode`** | Wrap a local model (HuggingFace, your own torch model). |
| **`DeterministicNode`** | Wrap a pure function — no model, just code. |

### Built-in providers

| Provider | Use it for | Auth |
|---|---|---|
| `stub` | Offline work, tests, demos (default) | none |
| `anthropic` | Claude models (Sonnet 4.5, Sonnet 5, Opus 5) | `ANTHROPIC_API_KEY` |
| `openai` | GPT-5 and any OpenAI-compatible API (Together, Groq, OpenRouter) | `OPENAI_API_KEY` |
| `ollama` | Local models (Llama 4 Scout, etc.) | none, runs on localhost:11434 |

### Run the test suite

```bash
git clone https://github.com/lordxmen2k/SNAIL-DSL.git
cd SNAIL-DSL
pip install -e ".[dev]"
pytest -q
```

91 tests pass.

---

## Installation (one-liner recap)

```bash
pip install snail-dsl
```

**Python:** 3.10, 3.11, 3.12

**Dependencies:** `pydantic>=2.0`, `httpx>=0.25,<1`, `pyyaml>=6.0,<7`, `click>=8.1,<9`

## Status

v0.2.1 — beta. The discipline is locked. See `CHANGELOG.md` for what changed since v0.1.0.

## License

Apache License 2.0. Copyright 2026 Tico Internet LLC.

## Links

- **PyPI:** https://pypi.org/project/snail-dsl/
- **Repository:** https://github.com/lordxmen2k/SNAIL-DSL
- **Issues:** https://github.com/lordxmen2k/SNAIL-DSL/issues
- **Book DOI:** `10.5281/zenodo.22839893`
