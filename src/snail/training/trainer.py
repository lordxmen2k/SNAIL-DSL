"""train_recipe — run a recipe as a SNAIL training loop.

For each node spec in the recipe, the trainer:
1. Iterates the dataset
2. Calls the user-provided `model_forward(payload, node_name)` for each sample
3. Writes a per-node trace file
4. Emits a frozen weights file at the recipe's `frozen_output` path

The trainer does not ship a real training algorithm — that's left to
the user's `model_forward`. v0.2.0 emits metadata + traces so the
discipline is in place: every training run produces a manifest.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from snail.training.recipe import Recipe


def _emit_weights_file(
    out_path: Path,
    *,
    recipe_name: str,
    node_name: str,
    distribution: str,
    weight_pin: str,
    epochs: int,
    learning_rate: float,
    samples_seen: int,
    final_loss: float,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format": "snail-json-v1",
        "recipe": recipe_name,
        "node": node_name,
        "distribution": distribution,
        "weight_pin": weight_pin,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "samples_seen": samples_seen,
        "final_loss": final_loss,
        "frozen_at": time.time(),
        "model_state": {"stub": True},
    }
    out_path.write_text(json.dumps(body, indent=2))
    return out_path


def train_recipe(
    recipe: Recipe,
    model_forward: Callable[[dict[str, Any], str], dict[str, Any]],
    dataset: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Run a recipe as a training loop.

    `model_forward(payload, node_name)` is called once per (sample, node)
    and must return a dict with at least a `confidence` key (used to
    compute loss = 1 - confidence).

    Returns a dict mapping node_name → path to its frozen weights file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = list(dataset)

    frozen_paths: dict[str, str] = {}
    for spec in recipe.node_specs:
        trace: list[dict[str, Any]] = []
        losses: list[float] = []
        for sample in samples:
            raw = model_forward(sample, spec.name)
            trace.append({"input": sample, "output": raw})
            confidence = float(raw.get("confidence", 1.0))
            losses.append(1.0 - confidence)

        # Write a per-node trace
        trace_path = out_dir / f"{spec.name}_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2))

        # Emit frozen weights metadata
        out_path = Path(recipe.frozen_output)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
        _emit_weights_file(
            out_path,
            recipe_name=recipe.name,
            node_name=spec.name,
            distribution=spec.distribution,
            weight_pin=spec.weight_pin,
            epochs=spec.epochs,
            learning_rate=spec.learning_rate,
            samples_seen=len(samples),
            final_loss=(sum(losses) / len(losses)) if losses else 0.0,
        )
        frozen_paths[spec.name] = str(out_path)

    return frozen_paths
