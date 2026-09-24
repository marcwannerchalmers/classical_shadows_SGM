"""Circuit helpers for GHZ-style and uniformly random Clifford operations."""

from typing import Any, Callable

from flax import struct
import jax
from jax import Array, jit, random, vmap
from jax import numpy as jnp
from jax.lax import cond, scan
import pennylane as qp


def build_parallel_entangler_blocks(
    selective_block: Array,
    n: int,
    xy_index: Array,
) -> None:
    """Apply a GHZ-style circuit to the selected qubits."""

    sorted_indices = jnp.argsort(
        selective_block, descending=True
    ).astype(jnp.int32)
    sorted_values = selective_block[sorted_indices]
    theta = jnp.asarray(jnp.pi / 2, dtype=jnp.float32)

    for index in range(n - 2, -1, -1):
        if (sorted_values[index] == 1) & (sorted_values[index + 1] == 1):
            qp.CNOT(
                jnp.stack([sorted_indices[index], sorted_indices[index + 1]])
            )
        if (index == 0) & (sorted_values[index] == 1):
            if xy_index == 0:
                qp.RX(theta, sorted_indices[index])
            else:
                qp.RY(theta, sorted_indices[index])


@struct.dataclass
class PartialCircuit:
    circuit_fun: Callable = struct.field(pytree_node=False)
    static_args: dict = struct.field(pytree_node=False)
    dynamic_args: dict

    def __call__(self, index) -> Any:
        self.circuit_fun(index, **self.static_args, **self.dynamic_args)


def sample_mallow(key: Array, n: int) -> tuple[Array, Array]:
    """Sample the Hadamard flags and permutation from the Mallow law."""

    available = jnp.arange(1, n + 1, dtype=jnp.int32)
    remaining = jnp.asarray(n, dtype=jnp.int32)

    def sample_step(carry, sample_key):
        available, remaining = carry
        size = available.shape[0]
        elements = jnp.arange(2 * size, dtype=jnp.int32)
        h_values = jnp.tile(
            jnp.asarray([[0], [1]], dtype=jnp.int32),
            (1, size),
        )
        k_values = jnp.tile(
            jnp.arange(1, size + 1, dtype=jnp.int32),
            (2, 1),
        )
        mask = (
            jnp.tile(jnp.arange(size, dtype=jnp.int32), (2, 1)) < remaining
        ).astype(jnp.int32)
        probabilities = jnp.asarray(
            2
            ** (
                remaining
                - 1
                + h_values
                + (remaining - k_values) * (-1) ** (1 + h_values)
            )
            / (4**remaining - 1),
            dtype=jnp.float32,
        ) * mask
        sample = random.choice(sample_key, elements, p=probabilities.flatten())
        h_value = sample // size
        k_value = sample % size
        maxima = jnp.sort(available, stable=False, descending=True)
        permutation_value = maxima[k_value]
        available = available.at[permutation_value - 1].set(0)
        return (
            available,
            remaining - 1,
        ), jnp.asarray([h_value, permutation_value], dtype=jnp.int32)

    _, values = scan(
        sample_step,
        init=(available, remaining),
        xs=random.split(key, n),
    )
    return values[:, 0], values[:, 1]


@jit
def _condition_gamma(
    h_i: Array,
    h_j: Array,
    permutation_i: Array,
    permutation_j: Array,
    value: Array,
) -> Array:
    keep = (
        ((h_i == 1) & (h_j == 1))
        | ((h_i == 1) & (h_j == 0) & (permutation_i < permutation_j))
        | ((h_i == 0) & (h_j == 1) & (permutation_i > permutation_j))
    )
    return cond(keep, lambda: value, lambda: jnp.zeros_like(value))


@jit
def _condition_delta(
    h_i: Array,
    h_j: Array,
    permutation_i: Array,
    permutation_j: Array,
    value: Array,
) -> Array:
    keep = (
        ((h_i == 0) & (h_j == 1))
        | ((h_i == 1) & (h_j == 1) & (permutation_i > permutation_j))
        | ((h_i == 0) & (h_j == 0) & (permutation_i < permutation_j))
    )
    return cond(keep, lambda: value, lambda: jnp.zeros_like(value))


def create_tableau(key: Array, n: int):
    """Sample the canonical-form data of a uniform random Clifford."""

    keys = random.split(key, 5)
    delta, delta_dag, gamma, gamma_dag = [
        random.randint(sample_key, (n, n), 0, 2, dtype=jnp.int32)
        for sample_key in keys[:4]
    ]
    h, permutation = sample_mallow(keys[4], n)
    gamma = vmap(
        vmap(_condition_gamma, in_axes=(0, None, 0, None, 0)),
        in_axes=(None, 0, None, 0, 1),
    )(h, h, permutation, permutation, gamma)
    gamma = jnp.tril(gamma) + jnp.tril(gamma, k=-1).T
    gamma_dag = jnp.tril(gamma_dag) + jnp.tril(gamma_dag, k=-1).T

    delta = vmap(
        vmap(_condition_delta, in_axes=(0, None, 0, None, 0)),
        in_axes=(None, 0, None, 0, 1),
    )(h, h, permutation, permutation, delta)
    delta = jnp.tril(delta, k=-1) + jnp.eye(n, dtype=delta.dtype)
    delta_dag = jnp.tril(delta_dag, k=-1) + jnp.eye(
        n, dtype=delta.dtype
    )
    return gamma, delta, gamma_dag, delta_dag, h, permutation - 1


def _apply_f(pauli_indices: Array, gamma: Array, delta: Array) -> None:
    n = gamma.shape[0]
    for row in reversed(range(n)):
        for column in reversed(range(row)):
            if delta[row, column] == 1:
                qp.CNOT(wires=jnp.asarray([row, column], dtype=jnp.int32))
    for row in reversed(range(n)):
        for column in reversed(range(row)):
            if gamma[row, column] == 1:
                qp.CZ(wires=jnp.asarray([row, column], dtype=jnp.int32))
    for wire in range(n):
        if pauli_indices[wire] == 1:
            qp.X(wire)
        elif pauli_indices[wire] == 2:
            qp.Y(wire)
        elif pauli_indices[wire] == 3:
            qp.Z(wire)
        if gamma[wire, wire] == 1:
            qp.S(wires=wire)


def _apply_f_inverse(
    pauli_indices: Array, gamma: Array, delta: Array
) -> None:
    n = gamma.shape[0]
    for wire in reversed(range(n)):
        if gamma[wire, wire] == 1:
            qp.adjoint(qp.S)(wires=wire)
        if pauli_indices[wire] == 1:
            qp.X(wire)
        elif pauli_indices[wire] == 2:
            qp.Y(wire)
        elif pauli_indices[wire] == 3:
            qp.Z(wire)
    for row in range(n):
        for column in range(row):
            if gamma[row, column] == 1:
                qp.CZ(wires=jnp.asarray([row, column], dtype=jnp.int32))
    for row in range(n):
        for column in range(row):
            if delta[row, column] == 1:
                qp.CNOT(wires=jnp.asarray([row, column], dtype=jnp.int32))


def canonical_form(
    gamma: Array,
    delta: Array,
    gamma_dag: Array,
    delta_dag: Array,
    h: Array,
    pauli_indices: Array,
    swap_indices: Array,
) -> None:
    """Apply a Clifford in its canonical decomposition."""

    n = gamma.shape[0]
    _apply_f(pauli_indices, gamma_dag, delta_dag)
    for index in reversed(range(n)):
        if swap_indices[index, 0] != swap_indices[index, 1]:
            qp.SWAP(wires=swap_indices[index])
    for wire in range(n):
        if h[wire] == 1:
            qp.H(wire)
    _apply_f(jnp.zeros(n, dtype=jnp.int32), gamma, delta)


def canonical_form_rev(
    gamma: Array,
    delta: Array,
    gamma_dag: Array,
    delta_dag: Array,
    h: Array,
    pauli_indices: Array,
    swap_indices: Array,
) -> None:
    """Apply the inverse canonical Clifford decomposition."""

    n = gamma.shape[0]
    _apply_f_inverse(jnp.zeros(n, dtype=jnp.int32), gamma, delta)
    for wire in reversed(range(n)):
        if h[wire] == 1:
            qp.H(wire)
    for index in range(n):
        if swap_indices[index, 0] != swap_indices[index, 1]:
            qp.SWAP(wires=swap_indices[index])
    _apply_f_inverse(pauli_indices, gamma_dag, delta_dag)


@jit
def permutation_to_swaps(permutation: Array) -> Array:
    """Return swaps that transform a permutation into ascending order."""

    n = permutation.shape[0]

    def swap_value(index, carry):
        current, left, right = carry
        position = jnp.argmax(current == index).astype(current.dtype)
        left = left.at[index].set(jnp.asarray(index, dtype=current.dtype))
        right = right.at[index].set(position)
        index_value = current[index]
        position_value = current[position]
        current = current.at[index].set(position_value)
        current = current.at[position].set(index_value)
        return current, left, right

    swaps = jnp.zeros(n, dtype=permutation.dtype)
    _, left, right = jax.lax.fori_loop(
        0,
        n,
        swap_value,
        (permutation, swaps, swaps),
    )
    return jnp.stack([left, right], axis=1)
