from __future__ import annotations

from functools import lru_cache, partial
from itertools import combinations
from math import comb
from typing import Iterable, Literal, Sequence

import jax
from jax import Array, jit, random, vmap
from jax import numpy as jnp
import numpy as np
import pennylane as qp

from tools.observable import PauliObservable
from tools.clifford import Tableau
from tools.statevector import _apply_batched_pauli
from tools.state import State
from shadow import Shadow


class MajoranaState(State):
    """Random fixed-particle fermionic states in Jordan-Wigner encoding.

    Attributes:
        state_dm: Batched complex64 statevectors restricted to one particle
            number sector.
    """

    state_dm: Array

    @classmethod
    def init_random(
        cls,
        key: Array,
        N_state: int,
        n: int,
        n_particles: int | None = None,
    ) -> "MajoranaState":
        if n_particles is None:
            n_particles = n // 2
        if not 0 <= n_particles <= n:
            raise ValueError("n_particles must satisfy 0 <= n_particles <= n")

        keys = random.split(key, N_state)
        states = vmap(cls.sample_state, in_axes=(0, None, None))(
            keys, n, n_particles
        )
        return cls(state_dm=states)

    @classmethod
    def init(cls, statevecs: Array) -> MajoranaState:
        return cls(state_dm=jnp.asarray(statevecs, dtype=jnp.complex64))

    @staticmethod
    @partial(jit, static_argnums=(1, 2))
    def sample_state(key: Array, n: int, n_particles: int) -> Array:
        key_real, key_imag = random.split(key)
        amplitudes = (
            random.normal(key_real, (2**n,), dtype=jnp.float32)
            + 1j * random.normal(key_imag, (2**n,), dtype=jnp.float32)
        )

        basis = jnp.arange(2**n, dtype=jnp.uint32)
        bit_positions = jnp.arange(n, dtype=jnp.uint32)
        occupations = jnp.sum(
            (basis[:, None] >> bit_positions) & jnp.uint32(1), axis=1
        )
        amplitudes = jnp.where(
            occupations == n_particles,
            amplitudes,
            jnp.complex64(0),
        )
        return (amplitudes / jnp.linalg.norm(amplitudes)).astype(jnp.complex64)

    @property
    def n(self) -> int:
        return int(np.log2(self.state_dm.shape[-1]))

    def prepare_state(self) -> None:
        qp.StatePrep(self.state_dm, wires=range(self.n))


_PAULI_PRODUCTS = {
    (0, 0): (1, 0),
    (0, 1): (1, 1),
    (0, 2): (1, 2),
    (0, 3): (1, 3),
    (1, 0): (1, 1),
    (1, 1): (1, 0),
    (1, 2): (1j, 3),
    (1, 3): (-1j, 2),
    (2, 0): (1, 2),
    (2, 1): (-1j, 3),
    (2, 2): (1, 0),
    (2, 3): (1j, 1),
    (3, 0): (1, 3),
    (3, 1): (1j, 2),
    (3, 2): (-1j, 1),
    (3, 3): (1, 0),
}


def majorana_tuples(n: int, k: int) -> list[tuple[int, ...]]:
    """Enumerate the Majorana basis needed for a fermionic k-RDM.

    Args:
        n: Number of fermionic modes, represented by qubits.
        k: Maximum RDM order.

    Returns:
        Strictly increasing Majorana-index tuples of even degrees `2` through
        `2 * k`.

    Raises:
        ValueError: If `k` is outside `1 <= k <= n`.
    """

    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")
    return [
        indices
        for degree in range(2, 2 * k + 1, 2)
        for indices in combinations(range(2 * n), degree)
    ]


def jordan_wigner_majorana(
    n: int, indices: Sequence[int]
) -> tuple[tuple[int, ...], int]:
    """Map an even Majorana product to its Jordan-Wigner Pauli word.

    Args:
        n: Number of fermionic modes.
        indices: Unique, strictly increasing Majorana indices.

    Returns:
        The integer-encoded Pauli word and its real unit sign.

    Raises:
        ValueError: If the degree is zero or odd, indices are not ordered and
            unique, or an index is outside `0 <= index < 2 * n`.
        RuntimeError: If the computed Jordan-Wigner phase is not real and unit
            magnitude.

    Notes:
        Pauli codes are `I=0`, `X=1`, `Y=2`, and `Z=3`.
    """

    if len(indices) == 0 or len(indices) % 2:
        raise ValueError("A physical Majorana observable needs positive even degree")
    if tuple(indices) != tuple(sorted(indices)) or len(set(indices)) != len(indices):
        raise ValueError("Majorana indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] >= 2 * n:
        raise ValueError("Majorana index outside 0, ..., 2*n-1")

    word = [0] * n
    phase = (-1j) ** (len(indices) // 2)
    for index in indices:
        mode = index // 2
        gamma = [3] * mode + [1 if index % 2 == 0 else 2]
        gamma += [0] * (n - mode - 1)
        for wire, right_pauli in enumerate(gamma):
            local_phase, word[wire] = _PAULI_PRODUCTS[
                (word[wire], right_pauli)
            ]
            phase *= local_phase

    if not np.isclose(phase.imag, 0.0) or not np.isclose(abs(phase.real), 1.0):
        raise RuntimeError("Jordan-Wigner image did not have a real unit phase")
    return tuple(word), 1 if phase.real > 0 else -1


def majorana_pauli_data(
    n: int, k: int
) -> tuple[Array, Array, list[tuple[int, ...]]]:
    """Build Pauli words, signs, and labels for a fermionic k-RDM.

    Args:
        n: Number of fermionic modes.
        k: Maximum RDM order.

    Returns:
        Pauli-code rows, Jordan-Wigner signs, and their Majorana-index labels.
    """

    labels = majorana_tuples(n, k)
    mapped = [jordan_wigner_majorana(n, indices) for indices in labels]
    params, signs = zip(*mapped)
    return (
        jnp.asarray(params, dtype=jnp.int32),
        jnp.asarray(signs, dtype=jnp.int32),
        labels,
    )


def majorana_observables_list(
    n_list: Iterable[int],
    k: int,
    subset: Literal["all", "random"] = "all",
    count: int = -1,
    key: Array | None = None,
) -> list[PauliObservable]:
    """Create Jordan-Wigner Pauli observables for each qubit count.

    Args:
        n_list: Qubit counts to enumerate.
        k: Maximum RDM order.
        subset: Whether to retain all rows or sample a random subset.
        count: Requested subset size. Values at least as large as the complete
            family retain every row.
        key: JAX random key required for random subsets.

    Returns:
        One observable batch per qubit count.

    Raises:
        AssertionError: If a random subset needs sampling but `count` is not
            positive or `key` is absent.
        ValueError: If `k` is invalid for a qubit count.
    """

    obs_list = []
    for n in n_list:
        data = majorana_pauli_data(n, k)[0]
        if subset == "random" and count < data.shape[0]:
            assert (count > 0) and (key is not None), "Invalid count"
            data = random.choice(key, data, shape=(count,), axis=0)
        
        obs_list.append(PauliObservable.init(data))

    return obs_list


def sample_even_majorana_permutations(
    key: Array, n: int, count: int = 1
) -> Array:
    """Sample uniform permutations from the alternating group `Alt(2*n)`.

    Args:
        key: JAX random key.
        n: Number of fermionic modes.
        count: Number of permutations.

    Returns:
        Permutation rows with shape `(count, 2 * n)`.

    Raises:
        ValueError: If `n` or `count` is not positive.
    """

    if n < 1 or count < 1:
        raise ValueError("n and count must be positive")

    keys = random.split(key, count)

    def sample_one(sample_key):
        permutation = random.permutation(
            sample_key, jnp.arange(2 * n, dtype=jnp.int32)
        )
        # inversions: number of pairs not in ascending order
        # count them once (triu) and sum
        inversions = jnp.sum(
            jnp.triu(permutation[:, None] > permutation[None, :], k=1),
            dtype=jnp.int32,
        )

        # swap last two elements to make it even if it is odd
        def make_even(value):
            return value.at[-2].set(value[-1]).at[-1].set(value[-2])

        return jax.lax.cond(inversions % 2 == 1, make_even, lambda x: x, permutation)

    return vmap(sample_one)(keys)

_PAULI_PRODUCT_PHASES = jnp.asarray(
    [
        [0, 0, 0, 0],
        [0, 0, 1, 3],
        [0, 3, 0, 1],
        [0, 1, 3, 0],
    ],
    dtype=jnp.int32,
)
_PAULI_PRODUCT_RESULTS = jnp.asarray(
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=jnp.int32,
)


def pauli_to_majorana_data(params: Array) -> tuple[Array, Array, Array]:
    """Recover Majorana data from Jordan-Wigner Pauli words.

    Args:
        params: Integer Pauli codes with the qubit axis last.

    Returns:
        The binary Majorana mask, Majorana degree, and real
        Majorana-to-Pauli sign.
    """

    params = jnp.asarray(params, dtype=jnp.int32)
    n = params.shape[-1]
    x = (params == 1) | (params == 2)
    z = (params == 2) | (params == 3)
    suffix_after = jnp.concatenate(
        [
            jnp.mod(
                jnp.cumsum(x[..., :0:-1], axis=-1, dtype=jnp.int32),
                2,
            )[..., ::-1],
            jnp.zeros(params.shape[:-1] + (1,), dtype=jnp.int32),
        ],
        axis=-1,
    ).astype(bool)
    odd = z ^ suffix_after
    even = x ^ odd
    mask = jnp.stack([even, odd], axis=-1).reshape(params.shape[:-1] + (2 * n,))
    degree = jnp.sum(mask, axis=-1, dtype=jnp.int32)

    word = jnp.zeros_like(params)
    phase = jnp.mod(-(degree // 2), 4)
    for index in range(2 * n):
        mode = index // 2
        gamma = jnp.asarray(
            [3] * mode
            + [1 if index % 2 == 0 else 2]
            + [0] * (n - mode - 1),
            dtype=jnp.int32,
        )
        active = mask[..., index]
        local_phase = jnp.sum(
            _PAULI_PRODUCT_PHASES[word, gamma], axis=-1, dtype=jnp.int32
        )
        multiplied_word = _PAULI_PRODUCT_RESULTS[word, gamma]
        phase = jnp.where(active, jnp.mod(phase + local_phase, 4), phase)
        word = jnp.where(active[..., None], multiplied_word, word)

    sign = jnp.where(phase == 0, 1, -1).astype(jnp.int32)
    return mask, degree, sign


def majorana_degree_from_pauli(params: Array) -> Array:
    """Return the Majorana degree of Jordan-Wigner Pauli words.

    Args:
        params: Integer Pauli codes with the qubit axis last.

    Returns:
        The number of Majorana operators in each word.
    """

    return pauli_to_majorana_data(params)[1]


@lru_cache(None)
def _adjacent_swap_schedule(n_majoranas: int) -> tuple[int, ...]:
    return tuple(
        position
        for end in range(n_majoranas - 1, 0, -1)
        for position in range(end)
    )


def _permutation_to_adjacent_flags(permutation: Array) -> Array:
    sequence = permutation
    flags = []
    for position in _adjacent_swap_schedule(permutation.shape[0]):
        active = sequence[position] > sequence[position + 1]
        left = sequence[position]
        right = sequence[position + 1]
        sequence = sequence.at[position].set(jnp.where(active, right, left))
        sequence = sequence.at[position + 1].set(jnp.where(active, left, right))
        flags.append(active)
    return jnp.asarray(flags, dtype=jnp.int32)


def _signed_permutation_from_flags(
    flags: Array, n_majoranas: int
) -> tuple[Array, Array]:
    permutation = jnp.arange(n_majoranas, dtype=jnp.int32)
    signs = jnp.ones((n_majoranas,), dtype=jnp.int32)
    schedule = _adjacent_swap_schedule(n_majoranas)

    for position, active in zip(reversed(schedule), reversed(flags)):
        maps_left = permutation == position
        maps_right = permutation == position + 1
        permutation = jnp.where(
            active & maps_left,
            position + 1,
            jnp.where(active & maps_right, position, permutation),
        )
        signs = jnp.where(active & maps_left, -signs, signs)

    return permutation, signs


@lru_cache(None)
def _majorana_pair_tables(n: int) -> tuple[np.ndarray, np.ndarray]:
    n_majoranas = 2 * n
    words = np.zeros((n_majoranas, n_majoranas, n), dtype=np.int32)
    signs = np.ones((n_majoranas, n_majoranas), dtype=np.int32)
    for first in range(n_majoranas):
        for second in range(n_majoranas):
            if first == second:
                continue
            indices = tuple(sorted((first, second)))
            word, sign = jordan_wigner_majorana(n, indices)
            words[first, second] = word
            signs[first, second] = sign if first < second else -sign
    return words, signs


@jit
def _sample_fermionic_gaussian_statevector(
    key: Array,
    indices: Array,
    input_statevectors: Array,
) -> Array:
    n = input_statevectors.shape[-1].bit_length() - 1
    n_majoranas = 2 * n
    permutations = indices[..., :n_majoranas]
    majorana_signs = indices[..., n_majoranas : 2 * n_majoranas]
    inverse_permutations = jnp.argsort(permutations, axis=-1)

    source_first = inverse_permutations[..., 0::2]
    source_second = inverse_permutations[..., 1::2]
    source_first_sign = jnp.take_along_axis(
        majorana_signs, source_first, axis=-1
    )
    source_second_sign = jnp.take_along_axis(
        majorana_signs, source_second, axis=-1
    )

    pair_words, pair_signs = _majorana_pair_tables(n)
    pauli_words = jnp.asarray(pair_words)[source_first, source_second]
    observable_signs = (
        source_first_sign
        * source_second_sign
        * jnp.asarray(pair_signs)[source_first, source_second]
    )
    x = (pauli_words == 1) | (pauli_words == 2)
    z = (pauli_words == 2) | (pauli_words == 3)
    sign_bits = observable_signs < 0

    n_samples = indices.shape[1]
    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (input_statevectors.shape[0], n_samples, input_statevectors.shape[-1]),
    )
    basis = jnp.arange(1 << n, dtype=jnp.int32)
    shifts = jnp.arange(n - 1, -1, -1, dtype=jnp.int32)
    keys = random.split(key, n)
    outcomes = jnp.zeros(
        (input_statevectors.shape[0], n_samples, n), dtype=indices.dtype
    )

    def measure_one(mode, carry):
        statevectors, outcomes = carry
        pauli_statevectors = _apply_batched_pauli(
            statevectors,
            x[..., mode, :],
            z[..., mode, :],
            sign_bits[..., mode],
            basis,
            shifts,
        )
        expectations = jnp.clip(
            jnp.real(
                jnp.sum(
                    jnp.conj(statevectors) * pauli_statevectors, axis=-1
                )
            ),
            -1,
            1,
        )
        probability_zero = (1 + expectations) / 2
        outcome = (
            random.uniform(
                keys[mode],
                probability_zero.shape,
                dtype=probability_zero.dtype,
            )
            >= probability_zero
        ).astype(indices.dtype)
        eigenvalue = 1 - 2 * outcome.astype(expectations.dtype)
        denominator = jnp.sqrt(
            jnp.maximum(
                2 * (1 + eigenvalue * expectations),
                jnp.finfo(expectations.dtype).tiny,
            )
        )
        statevectors = (
            statevectors + eigenvalue[..., None] * pauli_statevectors
        ) / denominator[..., None]
        return statevectors, outcomes.at[..., mode].set(outcome)

    _, outcomes = jax.lax.fori_loop(
        0, n, measure_one, (statevectors, outcomes), unroll=False
    )
    return outcomes


def _apply_tableau_majorana_exchange(
    tableau: Tableau, position: int, inverse: bool, active: Array
) -> Tableau:
    wire = position // 2
    if position % 2 == 0:
        repetitions = 1 if inverse else 3
        for _ in range(repetitions):
            tableau = tableau.S(wire, active)
        return tableau

    tableau = tableau.H(wire, active)
    tableau = tableau.H(wire + 1, active)
    tableau = tableau.CNOT(wire, wire + 1, active)
    repetitions = 1 if inverse else 3
    for _ in range(repetitions):
        tableau = tableau.S(wire + 1, active)
    tableau = tableau.CNOT(wire, wire + 1, active)
    tableau = tableau.H(wire, active)
    return tableau.H(wire + 1, active)


def _apply_tableau_permutation(
    tableau: Tableau, flags: Array, inverse: bool
) -> Tableau:
    schedule = _adjacent_swap_schedule(2 * tableau.n)
    steps = (
        zip(schedule, flags)
        if inverse
        else zip(reversed(schedule), reversed(flags))
    )
    for position, active in steps:
        tableau = _apply_tableau_majorana_exchange(
            tableau, position, inverse, active == 1
        )
    return tableau


class FermionicGaussianShadow(Shadow):
    """Classical shadows from even permutations of Jordan-Wigner Majoranas.

    The protocol samples an element of `Alt(2*n)`, measures the resulting
    commuting Majorana pairs, and reconstructs even-degree observables with
    the fermionic Gaussian inverse channel.
    """

    @classmethod
    def sample_indices(
        cls,
        key: Array,
        shape: tuple[int, ...],
        sample_idx_range: Array,
        *args,
        **kwargs,
    ) -> Array:
        del sample_idx_range, args, kwargs
        n_samples, n = shape
        permutations = sample_even_majorana_permutations(key, n, n_samples)
        flags = vmap(_permutation_to_adjacent_flags)(permutations)
        actual_permutations, signs = vmap(
            _signed_permutation_from_flags, in_axes=(0, None)
        )(flags, 2 * n)
        return jnp.concatenate([actual_permutations, signs, flags], axis=-1)

    @staticmethod
    def get_U_params(**kwargs):
        return {"n": kwargs["n"]}, {}

    @staticmethod
    def U_fun(ind: Array, n: int) -> None:
        n_majoranas = 2 * n
        flags = ind[2 * n_majoranas :]
        schedule = _adjacent_swap_schedule(n_majoranas)
        for position, active in zip(reversed(schedule), reversed(flags)):
            if active == 1:
                if position % 2 == 0:
                    qp.RZ(-jnp.pi / 2, wires=position // 2)
                else:
                    qp.IsingXX(
                        -jnp.pi / 2,
                        wires=[position // 2, position // 2 + 1],
                    )

    def _sample_statevector(self, states: State, key: Array) -> Array:
        return _sample_fermionic_gaussian_statevector(
            key,
            self.indices,
            states.state_dm,
        )

    @classmethod
    def circuit_clifford(cls, ind: Array, tableau: Tableau) -> Tableau:
        n_majoranas = 2 * tableau.n
        flags = ind[2 * n_majoranas :]
        return _apply_tableau_permutation(tableau, flags, inverse=False)

    @classmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array) -> Tableau:
        tableau = Tableau.create(outcome.shape[-1])
        tableau = tableau.MultiPauli(outcome)
        n_majoranas = 2 * tableau.n
        flags = ind[2 * n_majoranas :]
        return _apply_tableau_permutation(tableau, flags, inverse=True)

    @classmethod
    def _inverse_channel(
        cls, n: int, inv_oc: Array, obs: PauliObservable
    ) -> Array:
        degree = majorana_degree_from_pauli(obs.params)
        inverse_eigenvalues = jnp.asarray(
            [comb(2 * n, 2 * order) / comb(n, order) for order in range(n + 1)],
            dtype=jnp.float32,
        )
        valid = degree % 2 == 0
        return jnp.where(
            valid,
            jnp.asarray(inv_oc, dtype=jnp.float32)
            * inverse_eigenvalues[degree // 2],
            jnp.nan,
        )
