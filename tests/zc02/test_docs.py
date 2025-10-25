from pathlib import Path


def test_policy_section_contains_objective_and_gradient():
    tex = Path("docs/zc_02.tex").read_text()
    assert "J(\\theta) = \\mathbb{E}_{s \\sim d^{\\pi_\\theta},\\, a \\sim \\pi_\\theta}" in tex
    assert "\\nabla_\\theta J(\\theta) = \\mathbb{E}_{s, a}" in tex


def test_value_section_and_figure_alignment():
    tex = Path("docs/zc_02.tex").read_text()
    assert "v^{\\pi}(s) = \\sum_a \\pi(a \\mid s) q^{\\pi}(s, a)" in tex
    assert "q^{\\pi}(s, a) = r(s, a) + \\gamma" in tex
    assert "\\input{zc02/figures/policy_value_alignment.tex}" in tex
    assert "Milestone 3 policy/value alignment" in tex

    figure = Path("docs/zc02/figures/policy_value_alignment.tex").read_text()
    assert "Milestone" in figure and "Policy evaluation" in figure
    assert "v^{\\pi}(s) = \\sum_a \\pi(a \\mid s) q^{\\pi}(s,a)" in figure
