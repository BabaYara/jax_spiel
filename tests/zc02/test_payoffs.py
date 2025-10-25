import sympy as sp

from jax_spiel.zc02.payoffs import expected_row_payoff, expected_column_payoff


def test_expected_row_payoff_matches_sympy_simplification():
    p, q = sp.symbols("p q")
    r_tl, r_tr, r_bl, r_br = sp.symbols("r_tl r_tr r_bl r_br")

    row_payoffs = ((r_tl, r_tr), (r_bl, r_br))

    helper_expr = expected_row_payoff(p, q, row_payoffs=row_payoffs)
    sympy_expr = sp.expand(
        p * q * r_tl
        + p * (1 - q) * r_tr
        + (1 - p) * q * r_bl
        + (1 - p) * (1 - q) * r_br
    )

    assert sp.simplify(helper_expr - sympy_expr) == 0


def test_expected_column_payoff_matches_sympy_simplification():
    p, q = sp.symbols("p q")
    c_tl, c_tr, c_bl, c_br = sp.symbols("c_tl c_tr c_bl c_br")

    column_payoffs = ((c_tl, c_tr), (c_bl, c_br))

    helper_expr = expected_column_payoff(p, q, column_payoffs=column_payoffs)
    sympy_expr = sp.expand(
        p * q * c_tl
        + p * (1 - q) * c_tr
        + (1 - p) * q * c_bl
        + (1 - p) * (1 - q) * c_br
    )

    assert sp.simplify(helper_expr - sympy_expr) == 0

