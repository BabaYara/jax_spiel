from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Iterable, List


@dataclasses.dataclass
class Milestone:
    """Represents a plan milestone with an ordered list of bullet items."""

    title: str
    bullets: List[str]


_MILESTONE_PATTERN = re.compile(r"^(\d+)\. \*\*(.*?)\*\*\s*$")
_BULLET_PATTERN = re.compile(r"^\s*-\s+(.*\S)\s*$")
_CODE_PATTERN = re.compile(r"`([^`]+)`")


def parse_milestones(markdown: str) -> List[Milestone]:
    """Parse milestone entries from a markdown string.

    Args:
        markdown: Raw contents of the plan markdown document.

    Returns:
        A list of :class:`Milestone` objects preserving order from the document.
    """

    lines = markdown.splitlines()
    milestones: List[Milestone] = []
    in_milestones = False

    for line in lines:
        if line.startswith("## ") and "Milestones" in line:
            in_milestones = True
            continue
        if in_milestones and line.startswith("## ") and "Milestones" not in line:
            break
        if not in_milestones:
            continue
        milestone_match = _MILESTONE_PATTERN.match(line)
        if milestone_match:
            title = milestone_match.group(2).strip()
            milestones.append(Milestone(title=title, bullets=[]))
            continue
        bullet_match = _BULLET_PATTERN.match(line)
        if bullet_match and milestones:
            milestones[-1].bullets.append(bullet_match.group(1).strip())

    return milestones


def _escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_escape(text: str) -> str:
    """Escape characters that have special meaning in LaTeX."""

    result: list[str] = []
    last_index = 0
    for match in _CODE_PATTERN.finditer(text):
        result.append(_escape_plain(text[last_index : match.start()]))
        code = _escape_plain(match.group(1))
        result.append(r"\texttt{" + code + "}")
        last_index = match.end()
    result.append(_escape_plain(text[last_index:]))
    return "".join(result)


def format_milestones_to_latex(milestones: Iterable[Milestone]) -> str:
    """Format milestones for inclusion in LaTeX."""

    lines = ["\\begin{enumerate}[leftmargin=*]"]
    for milestone in milestones:
        lines.append(f"  \\item \\textbf{{{_latex_escape(milestone.title)}}}")
        if milestone.bullets:
            lines.append("    \\begin{itemize}[leftmargin=*]")
            for bullet in milestone.bullets:
                lines.append(f"      \\item {_latex_escape(bullet)}")
            lines.append("    \\end{itemize}")
    lines.append("\\end{enumerate}")
    lines.append("")
    return "\n".join(lines)


def write_milestones_tex(plan_path: Path, output_path: Path) -> str:
    """Generate the milestones TeX fragment from the plan document."""

    markdown = plan_path.read_text(encoding="utf-8")
    milestones = parse_milestones(markdown)
    latex = format_milestones_to_latex(milestones)
    output_path.write_text(latex, encoding="utf-8")
    return latex


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate milestones LaTeX fragment")
    parser.add_argument(
        "plan",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "plan.md",
        help="Path to plan markdown file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "milestones.tex",
        help="Destination path for generated LaTeX fragment.",
    )
    args = parser.parse_args(argv)
    write_milestones_tex(args.plan, args.output)


if __name__ == "__main__":
    main()
