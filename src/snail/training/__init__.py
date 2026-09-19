"""Training tooling — recipes, training loop, freezer, golden tests.

Public API:
    Recipe          — declarative recipe spec
    NodeSpec        — per-node config inside a recipe
    GoldenCase      — single golden test case
    load_recipe     — load a Recipe from a .recipe.yaml file
    train_recipe    — run a recipe as a training loop (Task 8)
    freeze_weights  — emit frozen weights file (Task 9)
    write_golden    — emit a golden case (Task 9)
    verify_golden   — verify golden cases against a Program (Task 9)

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations

from snail.training.parser import load_recipe
from snail.training.recipe import GoldenCase, NodeSpec, Recipe
from snail.training.trainer import train_recipe
from snail.training.freezer import compute_sha256, freeze_weights
from snail.training.golden import GoldenFailure, verify_golden, write_golden

__all__ = [
    "Recipe",
    "NodeSpec",
    "GoldenCase",
    "load_recipe",
    "train_recipe",
    "freeze_weights",
    "compute_sha256",
    "write_golden",
    "verify_golden",
    "GoldenFailure",
]
