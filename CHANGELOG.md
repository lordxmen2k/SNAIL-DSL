# Changelog

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
