"""CLI — `snail` command line entry point.

Subcommands:
    run      — execute a SNAIL program from a .py file
    inspect  — print program structure
    render   — render the program DAG to SVG
    train    — run a recipe through the trainer
    verify   — verify golden test cases

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click


def _load_program(path: Path) -> Any:
    """Load a Python file and return its first Program instance."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from snail.program import Program

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Program):
            return obj
    raise click.ClickException(
        f"No Program instance found in {path}. "
        "Define one and assign it to a module-level variable."
    )


@click.group()
@click.version_option()
def cli() -> None:
    """SNAIL — Single Node Activated Inference Layer."""


@cli.command()
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--input", "input_json", required=True, help="JSON-encoded program input.")
@click.option("--manifest-out", type=click.Path(path_type=Path), default=None)
def run(program_path: Path, input_json: str, manifest_out: Path | None) -> None:
    """Run a SNAIL program once."""
    program = _load_program(program_path)
    payload = json.loads(input_json)
    result = program.run(payload)
    manifest = result.manifest
    if hasattr(manifest, "model_dump"):
        manifest_obj = manifest.model_dump()
    elif hasattr(manifest, "dict"):
        manifest_obj = manifest.dict()
    else:
        manifest_obj = manifest
    click.echo(json.dumps(manifest_obj, indent=2, default=str))
    if manifest_out is not None:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest_obj, indent=2, default=str))
        click.echo(f"\nManifest written to {manifest_out}", err=True)


@cli.command()
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect(program_path: Path) -> None:
    """Print a SNAIL program's structure."""
    program = _load_program(program_path)
    click.echo(f"Program: {program.name}")
    click.echo(f"  Nodes ({len(program._node_proxies)}):")
    for name, proxy in program._node_proxies.items():
        click.echo(f"    - {name} (distribution={proxy.distribution!r})")
    click.echo(f"  Edges ({len(program._validated_edges)}):")
    for e in program._validated_edges:
        click.echo(
            f"    - {e.source_node}.{e.source_variant} → "
            f"{e.target_node}.{e.target_field}"
        )


@cli.command(name="render")
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--title", default=None)
def render_cmd(program_path: Path, out_path: Path, title: str | None) -> None:
    """Render a SNAIL program to SVG."""
    from snail.render import render_program_svg  # local import for cold-start speed

    program = _load_program(program_path)
    svg = render_program_svg(program, title=title or program.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)
    click.echo(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


@cli.command()
@click.argument("recipe_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--dataset", "dataset_json", default=None, help="JSON array of samples.")
def train(
    recipe_path: Path, output_dir: Path, dataset_json: str | None
) -> None:
    """Train a recipe and emit frozen weights."""
    from snail.training import load_recipe, train_recipe

    recipe = load_recipe(recipe_path)
    if dataset_json:
        dataset = json.loads(dataset_json)
    else:
        dataset = [{"sample": i} for i in range(3)]

    def stub_forward(payload, node_name):
        return {"label": "ok", "confidence": 0.9}

    paths = train_recipe(recipe, stub_forward, dataset, output_dir)
    for node_name, p in paths.items():
        click.echo(f"  {node_name} → {p}")


@cli.command()
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--golden-dir", "golden_dir", required=True, type=click.Path(path_type=Path))
def verify(program_path: Path, golden_dir: Path) -> None:
    """Verify a SNAIL program against golden test cases."""
    from snail.training import verify_golden

    program = _load_program(program_path)
    failures = verify_golden(program, golden_dir)
    if failures:
        click.echo(f"{len(failures)} failure(s):", err=True)
        for f in failures:
            click.echo(f"  - {f.case_id}: {f.reason}", err=True)
        sys.exit(1)
    click.echo("0 failures — all golden cases passed.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
