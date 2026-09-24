from __future__ import annotations

import jax
from jax import Array
from jax import numpy as jnp
import pennylane as qp


def depolarizing_noise(p: float, key: Array, n: int) -> None:
    """Apply independently sampled single-qubit depolarizing errors.

    Args:
        p: Total probability of a non-identity error. The X, Y, and Z errors
            each receive probability `p / 3`.
        key: JAX random key.
        n: Number of qubits.

    Notes:
        This function emits PennyLane operations and must be called while a
        circuit is being constructed.
    """

    probabilities = jnp.asarray([1 - p, p / 3, p / 3, p / 3])

    errors = jax.random.choice(
        key,
        4,
        shape=(n,),
        p=probabilities,
    )

    for wire in range(n):
        error = errors[wire]
        angle = jnp.asarray(jnp.pi, dtype=jnp.float32)
        qp.RX(angle * (error == 1), wires=wire)
        qp.RY(angle * (error == 2), wires=wire)
        qp.RZ(angle * (error == 3), wires=wire)
