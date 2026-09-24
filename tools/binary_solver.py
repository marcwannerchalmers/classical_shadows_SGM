"""Binary linear algebra used by stabilizer tableaus."""

import jax
from jax import Array
from jax import numpy as jnp
from jax.lax import cond, fori_loop


def solve_binary(matrix: jax.Array, target: jax.Array) -> tuple[Array, Array]:
    """Solve `matrix @ solution = target` over GF(2).

    Args:
        matrix: Binary coefficient matrix with at least as many rows as
            columns.
        target: Binary right-hand side with one entry per row.

    Returns:
        A candidate binary solution and a scalar boolean indicating whether
        the system is consistent.
    """

    rows, columns = matrix.shape
    augmented = jnp.concatenate([matrix, target[:, None]], axis=1)

    def eliminate(column, carry):
        current, active_rows = carry
        candidates = current[:, column] * active_rows
        pivot = jnp.argmax(candidates).astype(jnp.int32)
        pivot = cond(
            candidates[pivot] == 1,
            lambda: pivot,
            lambda: column.astype(jnp.int32),
        )

        column_row = current[column]
        pivot_row = current[pivot]
        active_rows = active_rows.at[column].set(0)
        current = current.at[column].set(pivot_row)
        current = current.at[pivot].set(column_row)
        factors = current[:, column] * active_rows
        return (current + jnp.outer(factors, current[column])) % 2, active_rows

    augmented, _ = fori_loop(
        0,
        columns,
        eliminate,
        (augmented, jnp.ones_like(target)),
    )
    inconsistent = jnp.any(
        jnp.all(augmented[columns:, :-1] == 0, axis=1)
        & (augmented[columns:, -1] == 1)
    )

    def substitute(offset, solution):
        row = columns - 1 - offset
        contribution = jnp.sum(
            augmented[row, :-1] * solution,
            dtype=jnp.int32,
        ) % 2
        return solution.at[row].set(
            (augmented[row, -1] + contribution) % 2
        )

    solution = fori_loop(
        0,
        columns,
        substitute,
        jnp.zeros(columns, dtype=augmented.dtype),
    )
    return solution, ~inconsistent


def stabilizer_phase(
    matrix: Array,
    signs: Array,
    coefficients: Array,
    target: Array,
) -> Array:
    """Compute the binary phase of a stabilizer-generator combination.

    Args:
        matrix: Symplectic generator matrix, stacked as X and Z blocks.
        signs: Binary signs of the generators.
        coefficients: Binary selector for generators in the combination.
        target: Target symplectic Pauli vector.

    Returns:
        The binary sign exponent of the target Pauli.
    """

    _, n = matrix.shape
    x = matrix[:n]
    z = matrix[n:]
    phase = jnp.sum(coefficients * signs, dtype=jnp.int32)
    pair_counts = z.T @ x
    commutation_phase = jnp.sum(
        jnp.triu(
            jnp.outer(coefficients, coefficients) * (pair_counts % 2),
            k=1,
        ),
        dtype=jnp.int32,
    )
    generator_y = jnp.sum(
        jnp.diag(jnp.outer(coefficients, coefficients) * pair_counts),
        dtype=jnp.int32,
    )
    target_y = jnp.dot(target[:n], target[n:])
    return (phase + commutation_phase + (generator_y - target_y) // 2) % 2
