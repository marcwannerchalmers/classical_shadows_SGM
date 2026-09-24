"""State models used by the shadow simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from flax import struct
from jax import Array, jit, random, vmap
from jax import numpy as jnp
import numpy as np
import pennylane as qp

from tools.clifford import GHZ_type_state_clifford, Tableau
from tools.utils import build_parallel_entangler_blocks


class State(ABC, struct.PyTreeNode):
    """Abstract state representation consumed by shadow simulators."""

    @classmethod
    @abstractmethod
    def init_random(
        cls,
        key: Array,
        N_state: int,
        n: int,
        *args: Any,
        **kwargs: Any,
    ) -> "State":
        """Sample a batch of random states.

        Args:
            key: JAX random key.
            N_state: Number of independent states.
            n: Number of qubits.
            *args: State-specific positional options.
            **kwargs: State-specific keyword options.

        Returns:
            A batched state object.
        """

    @classmethod
    @abstractmethod
    def init(cls, *args: Any, **kwargs: Any) -> "State":
        """Construct a state from an explicit representation.

        Args:
            *args: State-specific positional data.
            **kwargs: State-specific keyword data.

        Returns:
            The constructed state object.
        """

    @abstractmethod
    def prepare_state(self) -> None:
        """Emit PennyLane operations that prepare the state."""

    def __call__(self) -> None:
        self.prepare_state()

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of qubits represented by the state."""


class HRState(State):
    """Haar-random pure states represented by complex64 statevectors.

    Attributes:
        state_dm: Batched statevectors with final dimension `2**n`.
    """

    state_dm: Array

    @classmethod
    def init_random(cls, key: Array, N_state: int, n: int) -> "HRState":
        keys = random.split(key, N_state)
        return cls(state_dm=vmap(cls.sample_state, in_axes=(0, None))(keys, n))

    @classmethod
    def init(cls, statevecs: Array) -> "HRState":
        return cls(state_dm=jnp.asarray(statevecs, dtype=jnp.complex64))

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def sample_state(key: Array, n: int) -> Array:
        real_key, imaginary_key = random.split(key)
        amplitudes = (
            random.normal(real_key, (2**n,), dtype=jnp.float32)
            + 1j * random.normal(
                imaginary_key, (2**n,), dtype=jnp.float32
            )
        )
        return (amplitudes / jnp.linalg.norm(amplitudes)).astype(jnp.complex64)

    @property
    def n(self) -> int:
        return int(np.log2(self.state_dm.shape[-1]))

    def prepare_state(self) -> None:
        qp.StatePrep(self.state_dm, wires=range(self.n))


class GHZType(State):
    """GHZ-like stabilizer states on selected subsets of qubits.

    Attributes:
        selective_block: Binary masks selecting entangled qubits.
        xy: Per-state binary setting choosing the X/Y preparation phase.
    """

    selective_block: Array
    xy: Array

    @classmethod
    def init_random(cls, key: Array, N_state: int, n: int) -> "GHZType":
        settings = random.randint(
            key,
            (N_state, n + 1),
            0,
            2,
            dtype=jnp.int32,
        )
        return vmap(cls.init)(settings[:, :n], settings[:, n])

    @classmethod
    def init(cls, selective_block: Array, xy: Array) -> "GHZType":
        return cls(
            selective_block=jnp.asarray(selective_block, dtype=jnp.int32),
            xy=jnp.asarray(xy, dtype=jnp.int32),
        )

    @property
    def n(self) -> int:
        return int(self.selective_block.shape[-1])

    def prepare_state(self) -> None:
        build_parallel_entangler_blocks(self.selective_block, self.n, self.xy)

    def clifford(self, tableau: Tableau) -> Tableau:
        return GHZ_type_state_clifford(
            self.selective_block,
            self.xy,
            tableau,
        )

def GHZ_Klocal(locality: int) -> type[GHZType]:
    """Create a GHZ state class with a fixed number of active qubits.

    Args:
        locality: Number of randomly positioned active qubits in each state.

    Returns:
        A dynamically defined `GHZType` subclass suitable for an experiment.

    Raises:
        ValueError: When a generated state's qubit count is smaller than the
            requested locality.
    """

    class KLocalGHZ(GHZType):
        @classmethod
        def init_random(cls, key: Array, N_state: int, n: int) -> State:
            if not 1 <= locality <= n:
                raise ValueError("GHZ locality must satisfy 1 <= locality <= n")
            key_blocks, key_xy = random.split(key)
            priorities = random.uniform(key_blocks, (N_state, n), dtype=jnp.float32)
            active = jnp.argsort(priorities, axis=1)[:, :locality]
            blocks = jnp.zeros((N_state, n), dtype=jnp.int32)
            blocks = blocks.at[jnp.arange(N_state)[:, None], active].set(1)
            xy = random.randint(key_xy, (N_state,), 0, 2, dtype=jnp.int32)
            return cls.init(blocks, xy)

    KLocalGHZ.__name__ = f"GHZ_{locality}local"
    return KLocalGHZ