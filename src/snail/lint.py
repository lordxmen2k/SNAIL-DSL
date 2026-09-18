"""SNAIL lint rule — catches direct imports of model SDKs outside wrappers.

If you import openai/anthropic/transformers/etc. at module level in a
file that isn't a SNAIL wrapper, that's a violation. All model access
must go through @node or one of the snail.wrappers factories.

This is a discipline check, not a style preference. The lint module
exposes a `check_source(path)` function for CI integration and a
`FORBIDDEN_TOP_LEVEL_IMPORTS` set for users to extend.
"""

from __future__ import annotations
import ast
from pathlib import Path

# These modules, if imported at module level outside snail.wrappers,
# indicate someone is calling a model directly without going through @node.
FORBIDDEN_TOP_LEVEL_IMPORTS: set[str] = {
    "openai",
    "anthropic",
    "transformers",
    "torch",            # torch at module level outside training scripts
    "tensorflow",
    "jax",
    "flax",
    "replicate",
    "huggingface_hub",
    "litellm",
    "ollama",
    "langchain",
    "langgraph",
    "crewai",
}


def check_source(path: str | Path) -> list[str]:
    """Return a list of violation messages for the given Python file.

    Empty list = clean.
    """
    p = Path(path)
    if not p.exists():
        return [f"file not found: {p}"]

    # Don't lint the wrappers module itself — that's where model SDKs are
    # allowed to be imported.
    if "snail/wrappers" in str(p) or "snail\\wrappers" in str(p):
        return []
    # Don't lint the lint module itself.
    if "snail/lint" in str(p):
        return []

    try:
        tree = ast.parse(p.read_text())
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    violations.append(
                        f"{p}:{node.lineno}: direct import of {alias.name!r}. "
                        "Model SDKs must go through @node or snail.wrappers."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    violations.append(
                        f"{p}:{node.lineno}: direct import from {node.module!r}. "
                        "Model SDKs must go through @node or snail.wrappers."
                    )
    return violations


def check_directory(root: str | Path, exclude_dirs: set[str] | None = None) -> list[str]:
    """Walk a directory tree and lint every .py file. Returns all violations."""
    exclude = exclude_dirs or {".venv", "venv", "build", "dist", "__pycache__", ".git"}
    root = Path(root)
    all_violations: list[str] = []
    for p in root.rglob("*.py"):
        if any(part in exclude for part in p.parts):
            continue
        all_violations.extend(check_source(p))
    return all_violations
