"""Utilities to render the policy/value alignment figure.

The output is a small LaTeX table that mirrors the inline equations used in
``docs/zc_02.tex``.  Keeping the generation logic in Python makes it simple to
refresh the figure when the critique feedback changes.
"""
from __future__ import annotations

from pathlib import Path

FIGURE_PATH = Path(__file__).with_suffix(".tex")

FIGURE_ROWS = (
    ("Core API", "Immutable TensorState updates", r"$s' = f(s, a)$"),
    ("Policy evaluation", r"Inline objective $J(\theta)$", r"$J(\theta) = \mathbb{E}_{s, a}[r(s, a)]$"),
    ("Value estimation", r"Temporal recursion $v^{\pi}(s)$", r"$v^{\pi}(s) = \sum_a \pi(a \mid s) q^{\pi}(s, a)$"),
)

FIGURE_METADATA = {
    "label": "fig:policy-value-alignment",
    "caption": (
        "Policy/value alignment between milestone 3 objectives and the inline "
        "mathematical statements that drive the critique response."
    ),
    "inline_math": [row[2] for row in FIGURE_ROWS],
}


def render_policy_value_alignment(output_path: Path | None = None) -> str:
    """Render the LaTeX table and write it to ``output_path``.

    Parameters
    ----------
    output_path:
        Optional override for the path that should receive the generated table.
        When omitted the canonical location next to this script is used.

    Returns
    -------
    str
        The rendered LaTeX table with a trailing newline.
    """

    path = Path(output_path) if output_path else FIGURE_PATH
    header = "\\begin{tabular}{lll}"
    rows = [
        "\\textbf{Milestone} & \\textbf{Focus} & \\textbf{Key relation} \\\\",
    ]
    for milestone, focus, relation in FIGURE_ROWS:
        rows.append(f"{milestone} & {focus} & {relation} \\\\")
    body = "\n".join(rows)
    footer = "\\end{tabular}"
    rendered = "\n".join((header, body, footer)) + "\n"
    path.write_text(rendered)
    return rendered


if __name__ == "__main__":
    render_policy_value_alignment()
