"""Observable representations used by shadow reconstruction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar, Sequence

from flax import struct
import jax
from jax import Array, jit, random
from jax import numpy as jnp
import pennylane as qp
from pennylane.operation import Operator


class Observable(ABC, struct.PyTreeNode):
    """Abstract batched observable representation.

    Attributes:
        params: Observable parameters with a leading batch shape.
        weight: Scalar coefficient for each observable in the batch.
    """

    params: Array
    weight: Array

    @classmethod
    def init(cls, params: Array, weight: Array | None = None) -> "Observable":
        """Construct an observable batch.

        Args:
            params: Observable parameters. The final axis describes one
                observable.
            weight: Optional coefficient array matching `params.shape[:-1]`.

        Returns:
            The constructed observable batch.

        Raises:
            ValueError: If the weight shape does not match the parameter batch.
        """

        params = jnp.asarray(params)
        if weight is None:
            weight = jnp.ones(params.shape[:-1])
        if weight.shape != params.shape[:-1]:
            raise ValueError("Observable weights must match the parameter batch")
        return cls(params, weight)

    @abstractmethod
    def op(self) -> Operator:
        """Build the PennyLane operator used for expectation values."""

    def circuit(self) -> None:
        """Apply any basis-change circuit needed before measurement."""

    @abstractmethod
    def trace(self) -> Array:
        """Return the weighted trace of each observable."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of qubits acted on by each observable."""


_QP_OBSERVABLES = (qp.Identity, qp.PauliX, qp.PauliY, qp.PauliZ)


class PauliObservable(Observable):
    """A batch of Pauli words encoded as I=0, X=1, Y=2, Z=3.

    Attributes:
        params: Integer Pauli codes with shape `(..., n)`.
        weight: Coefficient for each Pauli word.
    """

    ztype_obs_list: tuple = struct.field(pytree_node=False)
    pauli_list: ClassVar[tuple[str, ...]] = ("I", "X", "Y", "Z")
    matrix_list: ClassVar[Array] = jnp.asarray(
        [operator(0).matrix() for operator in _QP_OBSERVABLES],
        dtype=jnp.complex64,
    )

    @classmethod
    @partial(jit, static_argnums=(0,))
    def init(
        cls, params: Array, weight: Array | None = None
    ) -> "PauliObservable":
        """Construct Pauli observables from integer-encoded words.

        Args:
            params: Pauli codes with the qubit axis last.
            weight: Optional coefficient array matching the leading shape.

        Returns:
            The normalized JAX representation.

        Raises:
            ValueError: If the weight shape does not match the parameter batch.
        """

        params = jnp.asarray(params, dtype=jnp.int32)
        if weight is None:
            weight = jnp.ones(params.shape[:-1])
        if weight.shape != params.shape[:-1]:
            raise ValueError("Observable weights must match the parameter batch")
        return cls(params, weight, cls._ztype_operators(params.shape[-1]))

    @classmethod
    def init_random(
        cls,
        key: Array,
        sample_indices: Sequence[int] = range(4),
        n: int = 1,
        N: int = 1,
        k_local: int = 0,
        padding_indices: Sequence[int] = (0,),
        weight: Array | None = None,
    ) -> "PauliObservable":
        """Sample random Pauli words with optional locality padding.

        Args:
            key: JAX random key.
            sample_indices: Pauli codes used on the active positions.
            n: Number of qubits per word.
            N: Number of words.
            k_local: Number of active positions. Nonpositive values and values
                larger than `n` select all positions.
            padding_indices: Pauli codes used on inactive positions.
            weight: Optional coefficient for each sampled word.

        Returns:
            The sampled observable batch.
        """

        if k_local <= 0 or k_local > n:
            k_local = n
        sample_key, padding_key, permutation_key = random.split(key, 3)
        sample_indices = jnp.asarray(sample_indices, dtype=jnp.int32)
        choices = random.randint(
            sample_key,
            (N, k_local),
            0,
            len(sample_indices),
            dtype=jnp.int32,
        )
        params = sample_indices[choices]

        if k_local < n:
            padding_indices = jnp.asarray(padding_indices, dtype=jnp.int32)
            padding_choices = random.randint(
                padding_key,
                (N, n - k_local),
                0,
                len(padding_indices),
                dtype=jnp.int32,
            )
            params = jnp.concatenate([params, padding_indices[padding_choices]], axis=1)
            params = random.permutation(
                permutation_key,
                params,
                axis=1,
                independent=True,
            )

        return cls.init(params, weight)

    @staticmethod
    def _ztype_operators(n: int) -> tuple:
        products = [
            qp.prod(*[qp.PauliZ(wire) for wire in range(n - 1, start - 1, -1)])
            for start in range(n - 1, -1, -1)
        ]
        return tuple(reversed([qp.Identity(0), *products]))

    def op(self) -> Operator:
        """Build a Catalyst-compatible PennyLane operator."""

        identity_count = jax.nn.one_hot(
            self.params, 4, dtype=jnp.float32
        )[..., 0].sum(axis=-1, dtype=jnp.int32)
        coefficients = jax.nn.one_hot(
            identity_count, self.n + 1, dtype=jnp.float32
        )
        return qp.sum(
            *[
                coefficients[index] * operator
                for index, operator in enumerate(self.ztype_obs_list)
            ]
        )

    def circuit(self) -> None:
        """Rotate X/Y factors into the computational basis."""

        for wire in range(self.n):
            if self.params[wire] == 1:
                qp.Hadamard(wire)
            elif self.params[wire] == 2:
                qp.adjoint(qp.S)(wire)
                qp.Hadamard(wire)

        sorted_indices = jnp.argsort(self.params).astype(jnp.int32)
        sorted_params = self.params[sorted_indices]
        for wire, target in enumerate(sorted_indices):
            if wire != target and sorted_params[wire] == 0:
                qp.SWAP(wires=[wire, target])

    def trace(self) -> Array:
        """Return each Pauli word's weighted matrix trace."""

        matrices = self.matrix_list[self.params]
        local_traces = jnp.einsum("...bii->...b", matrices)
        return jnp.prod(local_traces, axis=-1) * self.weight

    @classmethod
    def obs_string(cls, pauli_array: Array) -> str:
        """Convert an encoded Pauli word to an `IXYZ` string.

        Args:
            pauli_array: One-dimensional sequence of Pauli codes.

        Returns:
            The corresponding Pauli label string.
        """

        return "".join(cls.pauli_list[int(value)] for value in pauli_array)

    @property
    def n(self) -> int:
        """Number of qubits in each Pauli word."""

        return self.params.shape[-1]

    @property
    def is_ZType(self) -> Array:
        """Whether the current word contains only identity and Z factors."""

        return ~jnp.isin(
            jnp.asarray([1, 2], dtype=jnp.int32), self.params
        ).any()
