from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_SCRIPT = REPO_ROOT / "docs" / "zc02" / "figures" / "policy_value_alignment.py"
DOC_PATH = REPO_ROOT / "docs" / "zc_02.tex"


def _load_module():
    spec = importlib.util.spec_from_file_location("_policy_value_alignment", FIGURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_policy_value_alignment_render(tmp_path):
    module = _load_module()
    output_path = tmp_path / "policy_value_alignment.tex"
    rendered = module.render_policy_value_alignment(output_path)

    assert output_path.read_text() == rendered
    assert module.FIGURE_PATH.read_text() == rendered
    assert rendered.startswith("\\begin{tabular}")
    assert rendered.rstrip().endswith("\\end{tabular}")
    for inline in module.FIGURE_METADATA["inline_math"]:
        assert inline in rendered


def test_metadata_and_doc_alignment():
    module = _load_module()
    metadata = module.FIGURE_METADATA

    assert metadata["label"] == "fig:policy-value-alignment"
    assert "Policy/value alignment" in metadata["caption"]
    assert metadata["inline_math"], "Inline math relations should be declared"

    tex_source = module.FIGURE_PATH.read_text()
    assert tex_source.count("$") >= 6, "Each cell should keep inline math intact"

    doc_text = DOC_PATH.read_text()
    assert metadata["caption"] in doc_text
    assert metadata["label"] in doc_text
    assert "policy_value_alignment.py" in doc_text or "policy\\_value\\_alignment.py" in doc_text
    for inline in metadata["inline_math"]:
        normalized = inline.replace("\\\\", "\\").strip("$")
        lhs = normalized.split("=")[0].strip()
        assert lhs in doc_text
