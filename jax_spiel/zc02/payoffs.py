"""Symbolic helper functions for ZC02 payoff derivations."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

Scalar = Any

PayoffMatrix = Tuple[Tuple[Scalar, Scalar], Tuple[Scalar, Scalar]]


def _validate_payoff_matrix(payoffs: Sequence[Sequence[Scalar]]) -> PayoffMatrix:
    """Validates that *payoffs* behaves like a 2x2 payoff matrix."""

    if len(payoffs) != 2:
        raise ValueError("payoffs must have two rows")

    row0 = tuple(payoffs[0])
    row1 = tuple(payoffs[1])

    if len(row0) != 2 or len(row1) != 2:
        raise ValueError("each row of payoffs must have two entries")

    return (row0, row1)


def expected_row_payoff(
    p_top: Scalar,
    q_left: Scalar,
    *,
    row_payoffs: Sequence[Sequence[Scalar]],
) -> Scalar:
    """Computes the row player's expected payoff for a 2x2 matrix game."""

    (r_tl, r_tr), (r_bl, r_br) = _validate_payoff_matrix(row_payoffs)

    return (
        p_top * q_left * r_tl
        + p_top * (1 - q_left) * r_tr
        + (1 - p_top) * q_left * r_bl
        + (1 - p_top) * (1 - q_left) * r_br
    )


def expected_column_payoff(
    p_top: Scalar,
    q_left: Scalar,
    *,
    column_payoffs: Sequence[Sequence[Scalar]],
) -> Scalar:
    """Computes the column player's expected payoff for a 2x2 matrix game."""

    (c_tl, c_tr), (c_bl, c_br) = _validate_payoff_matrix(column_payoffs)

    return (
        p_top * q_left * c_tl
        + p_top * (1 - q_left) * c_tr
        + (1 - p_top) * q_left * c_bl
        + (1 - p_top) * (1 - q_left) * c_br
    )


__all__ = ["expected_row_payoff", "expected_column_payoff"]
