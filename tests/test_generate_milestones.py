from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "docs" / "zc02" / "generate_milestones.py"
    spec = importlib.util.spec_from_file_location("generate_milestones", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_milestones_matches_plan_snapshot():
    module = _load_module()
    plan_text = Path("docs/plan.md").read_text(encoding="utf-8")
    milestones = module.parse_milestones(plan_text)

    assert [m.title for m in milestones] == [
        "Core API Skeleton",
        "Example Game Implementations",
        "Policy & Value Functions",
        "Benchmarking & Testing",
    ]
    assert milestones[0].bullets == [
        "Define JAX-compatible `Game` and `State` abstractions with immutable dataclasses.",
        "Provide utilities for batched operations and JIT compilation.",
    ]


def test_format_milestones_includes_latex_environments():
    module = _load_module()
    plan_text = Path("docs/plan.md").read_text(encoding="utf-8")
    milestones = module.parse_milestones(plan_text)
    latex = module.format_milestones_to_latex(milestones)

    assert "\\begin{enumerate}" in latex
    assert "\\texttt{Game}" in latex
    assert "\\textbf{Core API Skeleton}" in latex
