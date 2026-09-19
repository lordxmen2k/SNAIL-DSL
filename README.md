# SNAIL — Single Node Activated Inference Layer

[![PyPI version](https://img.shields.io/pypi/v/snail-dsl.svg)](https://pypi.org/project/snail-dsl/)
[![Python versions](https://img.shields.io/pypi/pyversions/snail-dsl.svg)](https://pypi.org/project/snail-dsl/#files)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lordxmen2k/SNAIL-DSL/blob/main/LICENSE)
[![Status: Beta](https://img.shields.io/pypi/status/snail-dsl.svg)](https://pypi.org/project/snail-dsl/)
[![Downloads](https://img.shields.io/pypi/dm/snail-dsl.svg)](https://pypistats.org/packages/snail-dsl)

**A Python DSL for composing frozen, single-pass neural primitives into statically-typed dataflow programs.**

SNAIL programs are *recipes* — declared in Python, compiled into a typed DAG at construction time, executed one forward pass at a time. Every node is frozen, every output is locked, and out-of-distribution inputs are a first-class type rather than a runtime crash.

```bash
pip install snail-dsl
```

**You do not need an API key to start.** SNAIL ships with a stub provider so you can build, run, and test programs without any external service.

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
- An API key for any LLM (we use the built-in stub).
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
Successfully installed annotated-types-0.8.0 click-8.5.0 httpx-0.28.1 pydantic-2.13.5 pyyaml-6.0.3 snail-dsl-0.2.0 ...
```

Now confirm the install worked:

```bash
python -c "import snail; print(snail.__version__)"
```

Expected output:

```
0.2.0
```

If you see `0.2.0`, the install worked. Move to Step 2.

---

## ✅ Step 2 — Run your first SNAIL program

We are going to type a small program into a file, then run it. This program classifies the intent of a customer message. It has two nodes (two small functions) connected together.

### 2.1 — Create the program file

Open your text editor (Notepad on Windows, TextEdit on macOS, VS Code, anything). Create a new file called `hello_snail.py` inside your `snail-playground` folder.

Copy-paste this **exactly** into the file:

```python
"""hello_snail.py — your first SNAIL program."""

# 1. We need these types from SNAIL and pydantic.
from pydantic import BaseModel
from snail import (
    node,
    Program,
    edge,
    NodeContext,
    NodeResult,
    OODSignal,
)

# 2. Tell SNAIL what a "good" output looks like.
#    A node returns either `ok` (success) or `ood` (not sure).
class IntentOk(BaseModel):
    intent: str
    confidence: float

class Intent(NodeResult):
    ok: IntentOk | None = None
    ood: OODSignal | None = None

# 3. Wrap a function with @node. This becomes a "node" in the DAG.
@node(
    name="classify_intent",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3",   # the training distribution
    confidence_threshold=0.7,            # below this → OOD
)
def classify_intent(ctx: NodeContext, weights, message: dict):
    text = message.get("text", "").lower()
    if "refund" in text:
        intent = "refund"
    elif "balance" in text:
        intent = "billing"
    else:
        intent = "other"
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
    input_schema=Intent,
    output_schema=Entities,
    distribution="customer_entities_v3",
    confidence_threshold=0.6,
)
def extract_entities(ctx: NodeContext, weights, intent_result: Intent):
    if intent_result.is_ood:
        return Entities(ood=intent_result.ood)
    return Entities(ok=EntitiesOk(order_id="ORD-12345", confidence=0.85))

# 5. Wire them into a program (the typed DAG).
program = Program(
    name="hello_snail",
    nodes=[classify_intent, extract_entities],
    edges=[edge(classify_intent.ok)],   # OK output of classify → input of extract
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
- One edge connects them: the OK output of `classify_intent` feeds into the input of `extract_entities`.

---

## 🎨 Step 4 — Render the DAG to SVG

Another subcommand draws your program as a picture:

```bash
snail render hello_snail.py --out hello_snail.svg
```

Open `hello_snail.svg` in any web browser. You'll see a picture with two boxes (your nodes) and a line between them (the edge). The boxes have rounded corners and the background is dark navy — that is the SNAIL visual style.

---

## 🚂 Step 5 — Train a recipe (no real model needed)

Now we teach SNAIL how to "train" a node from a recipe. Recipes are YAML files that say "here is the dataset, here are the training settings, here is where to put the frozen weights when done."

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

This is the **frozen weights file**. It records exactly what training happened. You can audit it, ship it, pin a specific version, or reject any program that uses a different version.

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

For Anthropic:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (Git Bash)
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

For OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

For Ollama (local, no key needed):

```bash
# Install Ollama from https://ollama.com, then pull a model
ollama pull llama4-scout
```

### 7.2 — Switch the provider in your code

In `hello_snail.py`, find the line where you built `summarize` (or add one). Replace `provider="stub"` with one of:

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

You have now used every piece of SNAIL v0.2.0:

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

v0.2.0 — beta. The discipline is locked. See `CHANGELOG.md` for what changed since v0.1.0.

## License

Apache License 2.0. Copyright 2026 Tico Internet LLC.

## Links

- **PyPI:** https://pypi.org/project/snail-dsl/
- **Repository:** https://github.com/lordxmen2k/SNAIL-DSL
- **Issues:** https://github.com/lordxmen2k/SNAIL-DSL/issues
- **Book DOI:** `10.5281/zenodo.22839893`
