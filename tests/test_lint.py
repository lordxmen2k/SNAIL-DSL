"""Tests for the lint rule that catches direct model SDK imports."""

import os
import tempfile
from pathlib import Path
from snail.lint import check_source, check_directory, FORBIDDEN_TOP_LEVEL_IMPORTS


def test_lint_clean_file(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text("from snail import node, Program\nimport json\n")
    assert check_source(p) == []


def test_lint_catches_openai_import(tmp_path):
    p = tmp_path / "dirty.py"
    p.write_text("import openai\nclient = openai.Client()\n")
    violations = check_source(p)
    assert len(violations) == 1
    assert "openai" in violations[0]


def test_lint_catches_anthropic_import_from(tmp_path):
    p = tmp_path / "dirty.py"
    p.write_text("from anthropic import Anthropic\n")
    violations = check_source(p)
    assert len(violations) == 1
    assert "anthropic" in violations[0]


def test_lint_catches_torch_import(tmp_path):
    p = tmp_path / "dirty.py"
    p.write_text("import torch\n")
    violations = check_source(p)
    assert len(violations) == 1
    assert "torch" in violations[0]


def test_lint_ignores_wrappers_module(tmp_path):
    """snail/wrappers is allowed to import model SDKs."""
    p = tmp_path / "snail" / "wrappers" / "openai_wrapper.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("import openai\n")
    violations = check_source(p)
    assert violations == []


def test_lint_directory_walk(tmp_path):
    (tmp_path / "good.py").write_text("from snail import node\n")
    (tmp_path / "bad.py").write_text("import transformers\n")
    violations = check_directory(tmp_path)
    assert len(violations) == 1
    assert "transformers" in violations[0]


def test_lint_excludes_dirs(tmp_path):
    (tmp_path / "ok.py").write_text("import openai\n")
    venv = tmp_path / ".venv" / "lib" / "site.py"
    venv.parent.mkdir(parents=True, exist_ok=True)
    venv.write_text("import openai\n")

    violations = check_directory(tmp_path, exclude_dirs={".venv"})
    assert len(violations) == 1
    assert "ok.py" in violations[0]


def test_lint_forbidden_set_is_well_known():
    """Sanity check: the forbidden set contains the SDKs we care about."""
    assert "openai" in FORBIDDEN_TOP_LEVEL_IMPORTS
    assert "anthropic" in FORBIDDEN_TOP_LEVEL_IMPORTS
    assert "transformers" in FORBIDDEN_TOP_LEVEL_IMPORTS
    assert "torch" in FORBIDDEN_TOP_LEVEL_IMPORTS
