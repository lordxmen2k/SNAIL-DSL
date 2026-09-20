# Changelog

## [0.4.1] — 2026-09-20

### Fixed
- **Program validation no longer chokes on ParallelGroup mixed with Edge in the validation loop.** The OK-edge `claimed_by_ok` loop now guards against non-Edge entries (e.g., `EscalationSpec`, `ParallelGroup`) which were incorrectly appearing in `_raw_edges` after the v0.3.0 changes. Affects programs that mix `escalations=[]` with `parallel_groups=[]` in the same `Program(...)` call.
- **`examples/end_to_end.py`**: a worked end-to-end demo that exercises `escalate`, `parallel_edges`, `weight_pin`, and `calibration` in one pipeline.

### Changed
- **README** now has a top-level "What's new in v0.3.0 + v0.4.0" section with code examples for every new primitive. PyPI's first impression matches the v0.4.0 reality.
- CLI commands table includes the new `snail calibrate` subcommand.

### Backward compatibility
- All 127 v0.4.0 tests still pass without modification.

## [0.4.0] — 2026-09-20

### Added
- **SHA-256 weight pinning** (`src/snail/weights.py`). `parse_weight_pin(pin)` validates the `<model>@sha256:<64-hex>` format; `verify_weight_pin(pin, path)` raises `WeightPinMismatch` if the file's hash doesn't match. `Program(...)` runs the check at construction time — a mismatched pin fails the program before `.run()` is called.
- **`@node` decorator accepts `weight_pin`** alongside `frozen_weights`. The pin survives into the SnailNode and is read by Program validation.
- **Confidence calibration tooling** (`src/snail/calibrate.py`). `run_calibration(program, golden_cases)` produces a `CalibrationReport` with ECE over 10 bins, per-bin statistics, and JSON serialization. `plot_reliability_diogram()` emits a matplotlib reliability diagram (optional).
- **`snail calibrate` CLI subcommand**. Loads a program, runs all golden cases under `--golden-dir`, writes JSON to `--output`, optionally writes a PNG to `--plot`. Exits 0 if ECE < `--threshold` (default 0.10); exits 1 otherwise. Suitable for CI gating.
- **Example: `examples/calibration_demo.py`** shows the calibration report shape end-to-end.

### Backward compatibility
- All 114 v0.3.0 tests pass without modification.
- The new `weight_pin` decorator kwarg is opt-in. Existing nodes without it behave exactly as in v0.3.0.

### Benchmarks (all green)
- 127 tests passing (114 v0.3.0 + 7 weight_pin + 6 calibrate).

## [0.3.0] — 2026-09-20

### Added
- **`escalate(node, to=other)` primitive** (`src/snail/escalation.py`). One-line wiring for the OOD → heavier-model pattern. Rejects cost-tier downgrades (`small → large` is allowed; `large → small` is not) and cycles at composition time. Returns an `EscalationSpec` consumed by `Program(escalations=[...])`.
- **`parallel_edges(inputs=node, to=[list])` primitive** (`src/snail/parallel.py`). Multiple nodes receive the same input; downstream merge receives each as a typed field. Rejects duplicates and empty lists.
- **`CostTier` enum** — small / medium / large. `HostedNode` and `DeterministicNode` accept a `cost_tier` kwarg (default `medium` / `small` respectively).
- **Cost/token accounting in `Manifest`**: every `ManifestNodeEvent` now carries `tokens_in`, `tokens_out`, `cost_usd`, `model_id`. `Manifest.summary` aggregates totals and adds `nodes_fired` + `escalations_triggered` counters.
- **`ProviderResponse.tokens_in/out/cost_usd` fields** populated from Anthropic `usage.input_tokens/output_tokens` and OpenAI-compat `usage.prompt_tokens/completion_tokens`.
- **`snail.providers.pricing`** — per-model USD pricing table for Anthropic (claude-haiku-4-5/5, claude-sonnet-4-5/5, claude-opus-4/4-5/5) and OpenAI (gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini). User-overridable via `cost_per_1k_input/output` kwargs.

### Changed
- `Program.__init__` accepts new kwargs: `escalations=[...]` and `parallel_groups=[...]`.
- Edge resolution: multiple OK edges can target the same downstream node when they go to different fields (enables parallel fan-out merge).
- `Manifest.to_dict()` and `Manifest.to_json()` include the new `summary` block.

### Backward compatibility
- All 91 v0.2.x tests pass without modification.
- `Manifest` schema is additive — old readers see the same fields they always did.

### Benchmarks (all green)
- 114 tests passing (91 v0.2.x + 6 escalation + 11 cost/manifest + 6 parallel).

## [0.2.3] — 2026-09-19

### Changed
- **Project description rewritten** to lead with the deterministic-pipeline positioning: "Deterministic LLM pipeline DSL — compose frozen, single-pass neural primitives into typed dataflow programs with OOD-as-type and tamper-evident manifests."
- **README opening rewritten** to position SNAIL as the alternative to agentic LLM frameworks (LangChain / Claude Agent SDK / AutoGPT) before the glossary section.
- **PyPI tags refreshed.** Dropped `agents`. Added `pipelines`, `workflows`, `deterministic-ai`, `typed-pipeline`, `frozen-weights`, `golden-tests`.

## [0.2.0] — 2026-09-19

### Added
- **Provider registry** under `snail.providers.*`. `Provider` protocol + `ProviderRegistry`. Built-in providers: `stub` (default, no-network), `anthropic` (Claude Messages API), `openai` (Chat Completions, also works for Together/Groq/OpenRouter), `ollama` (local `/api/chat`).
- **`HostedNode` dispatches through Provider registry.** New `provider` and `provider_kwargs` kwargs. Default provider is `stub` so v0.1.0 call sites keep working.
- **Training pipeline tooling** under `snail.training.*`. `Recipe` dataclass + YAML loader (`load_recipe`), `train_recipe`, `freeze_weights`, `compute_sha256`, `write_golden`, `verify_golden`, `GoldenFailure`. Sample recipe at `examples/recipe_classify_intent.recipe.yaml`.
- **DAG renderer** under `snail.render.*`. `layout_program` (layered layout, no overlap), `render_svg`, `render_program_svg` (one-call public API), dark theme matching the book cover.
- **CLI** (`snail` command, installed as a console script):
  - `snail run PROGRAM_PATH --input '{...}'` — run a program, emit manifest.
  - `snail inspect PROGRAM_PATH` — print program structure.
  - `snail render PROGRAM_PATH --out out.svg` — render SVG.
  - `snail train RECIPE_PATH --output-dir DIR` — run recipe training, emit weights.
  - `snail verify PROGRAM_PATH --golden-dir DIR` — run golden cases, exit 0/1.
- New dependencies: `click>=8.1,<9`, `httpx>=0.25,<1`, `pyyaml>=6.0,<7`.

### Changed
- `src/snail/wrappers.py`: `HostedNode` body now dispatches through `ProviderRegistry.get(provider).complete(...)` instead of the v0.1.0 inline stub.
- `src/snail/node.py`: `_extract_confidence` now handles `None` values gracefully.

### Backward compatibility
- All 41 v0.1.0 tests pass without modification.
- `HostedNode(...)` without the new `provider` kwarg still works — defaults to `stub`.

## [0.1.0] — 2026-09-12

- Initial release. Frozen nodes, typed DAG, OOD-as-type, manifest emission, lint rules.
