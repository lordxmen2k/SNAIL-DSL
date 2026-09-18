# SNAIL — Deploy Instructions

This document is the **exact recipe** to publish `snail-dsl` v0.1.0 to
**real PyPI** (`pypi.org`, NOT TestPyPI).

## What's in this package

```
snail-dsl/
├── dist/
│   ├── snail_dsl-0.1.0-py3-none-any.whl      ← built wheel
│   └── snail-dsl-0.1.0.tar.gz                 ← source distribution
├── src/snail/                                  ← package source
├── tests/                                      ← 41 pytest tests (all passing)
├── examples/                                   ← 5 working example programs
├── README.md
├── LICENSE
├── pyproject.toml
└── DEPLOY.md                                   ← you are here
```

## Test status

```
$ pytest
============================== 41 passed in 0.30s ==============================
```

The full test suite covers:

- `tests/test_node.py` — 10 tests for the @node decorator (frozen weights, single pass, OOD flipping, locked output)
- `tests/test_program.py` — 11 tests for Program / edge() / DAG composition / manifests
- `tests/test_wrappers.py` — 8 tests for ExternalLocalNode, HostedNode, DeterministicNode
- `tests/test_lint.py` — 8 tests for the discipline lint rule
- `tests/golden/test_extract_invoice_total.py` — 4 golden tests for a real node

## Deploy steps (run from your local machine)

### 1. Set up your PyPI API token

Get a token from https://pypi.org/manage/account/token/ (scope: "project: snail-dsl").

Configure `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-...your-token-here...
```

OR set the env var:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...your-token-here...
```

### 2. Upload to PyPI

From the package root (the directory containing `dist/`):

```bash
# Verify the artifacts look right
twine check dist/*

# Upload to REAL PyPI (no --repository flag, no TestPyPI)
twine upload dist/*
```

You should see output like:

```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading snail_dsl-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.5/12.5 kB • 00:00 • ?
Uploading snail-dsl-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.2/14.2 kB • 00:00 • ?
```

### 3. Verify

```bash
# From a fresh shell:
pip install snail-dsl
python -c "from snail import node, Program, edge; print('SNAIL live on PyPI')"
```

Then visit https://pypi.org/project/snail-dsl/ to confirm.

## What ships in v0.1.0

- ✅ `@node` decorator with frozen weights, single-pass enforcement, OOD-as-type, locked output
- ✅ `Program` typed DAG container with static validation (cycles, missing edges, unknown nodes)
- ✅ `edge()` builder with `.ok` / `.ood` accessors and auto-routing
- ✅ `Manifest` — structured per-run audit log (JSON-serializable)
- ✅ Three wrappers: `ExternalLocalNode`, `HostedNode`, `DetermisticNode`
- ✅ Lint rule that catches direct model SDK imports outside wrappers
- ✅ 41 passing tests
- ✅ 5 working example programs (invoice pipeline, ResNet classifier, email triage,
     customer support agent, ML training pipeline, data validation)
- ✅ Apache 2.0 license

## What does NOT ship in v0.1.0 (planned for v0.2.0+)

- Training pipeline tooling (synthetic data generation, model export to `.snail` format)
- Real provider clients for HostedNode (Anthropic, OpenAI, etc. — v0.1.0 uses a stub)
- CLI (`snail run program.py`)
- Visual DAG renderer

## What this is

A v0.1.0 alpha. The core primitives work and the discipline is enforced
in the type system. The next releases fill in the surrounding tooling
(training, providers, CLI). The book (planned for after this ships)
will walk through the architecture and use these primitives as the
foundation.

## Contact

Built by Tico Internet LLC. Apache 2.0. Issues → GitHub repo.
