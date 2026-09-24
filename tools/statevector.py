import numpy as np
import jax
from jax import numpy as jnp
from jax import random, jit, lax
from tools.clifford import computational_basis_measurement_paulis


def _apply_single_qubit_gates(statevectors, gates, wire, n: int):
    basis = jnp.arange(1 << n, dtype=jnp.int32)

    shift = (
        jnp.asarray(n - 1, dtype=jnp.int32)
        - jnp.asarray(wire, dtype=jnp.int32)
    )
    mask = jnp.left_shift(
        jnp.int32(1),
        shift,
    )

    zero_basis = jnp.bitwise_and(
        basis,
        jnp.bitwise_not(mask),
    )
    one_basis = jnp.bitwise_or(
        zero_basis,
        mask,
    )

    amp0 = jnp.take(
        statevectors,
        zero_basis,
        axis=-1,
    )
    amp1 = jnp.take(
        statevectors,
        one_basis,
        axis=-1,
    )

    out0 = (
        gates[..., 0, 0, None] * amp0
        + gates[..., 0, 1, None] * amp1
    )
    out1 = (
        gates[..., 1, 0, None] * amp0
        + gates[..., 1, 1, None] * amp1
    )

    bit_is_one = (
        jnp.bitwise_and(
            basis,
            mask,
        )
        != 0
    )

    return jnp.where(
        bit_is_one,
        out1,
        out0,
    )


def _apply_dynamic_single_qubit_gates(
    statevectors,
    gates,
    wires,
    n: int,
):
    basis = jnp.arange(
        1 << n,
        dtype=jnp.int32,
    )

    shifts = (
        jnp.asarray(
            n - 1,
            dtype=jnp.int32,
        )
        - wires.astype(jnp.int32)
    )

    masks = jnp.left_shift(
        jnp.ones_like(
            wires,
            dtype=jnp.int32,
        ),
        shifts,
    )

    zero_basis = jnp.bitwise_and(
        basis,
        jnp.bitwise_not(
            masks[..., None]
        ),
    )

    one_basis = jnp.bitwise_or(
        zero_basis,
        masks[..., None],
    )

    amp0 = jnp.take_along_axis(
        statevectors,
        zero_basis,
        axis=-1,
    )
    amp1 = jnp.take_along_axis(
        statevectors,
        one_basis,
        axis=-1,
    )

    out0 = (
        gates[..., 0, 0, None] * amp0
        + gates[..., 0, 1, None] * amp1
    )
    out1 = (
        gates[..., 1, 0, None] * amp0
        + gates[..., 1, 1, None] * amp1
    )

    bit_is_one = (
        jnp.bitwise_and(
            basis,
            masks[..., None],
        )
        != 0
    )

    return jnp.where(
        bit_is_one,
        out1,
        out0,
    )


def _apply_dynamic_cnot(
    statevectors,
    controls,
    targets,
    condition,
    n: int,
):
    basis = jnp.arange(
        1 << n,
        dtype=jnp.int32,
    )

    control_masks = jnp.left_shift(
        jnp.ones_like(
            controls,
            dtype=jnp.int32,
        ),
        (
            jnp.asarray(
                n - 1,
                dtype=jnp.int32,
            )
            - controls.astype(jnp.int32)
        ),
    )

    target_masks = jnp.left_shift(
        jnp.ones_like(
            targets,
            dtype=jnp.int32,
        ),
        (
            jnp.asarray(
                n - 1,
                dtype=jnp.int32,
            )
            - targets.astype(jnp.int32)
        ),
    )

    control_is_one = (
        jnp.bitwise_and(
            basis,
            control_masks[..., None],
        )
        != 0
    )

    permutation = jnp.bitwise_xor(
        basis,
        jnp.where(
            control_is_one,
            target_masks[..., None],
            jnp.int32(0),
        ),
    )

    transformed = jnp.take_along_axis(
        statevectors,
        permutation,
        axis=-1,
    )

    return jnp.where(
        condition[..., None],
        transformed,
        statevectors,
    )


def _single_qubit_gate_tables(dtype):
    inv_sqrt_two = jnp.asarray(
        1 / np.sqrt(2),
        dtype=dtype,
    )

    identity = jnp.eye(
        2,
        dtype=dtype,
    )

    hadamard = (
        inv_sqrt_two
        * jnp.asarray(
            [
                [1, 1],
                [1, -1],
            ],
            dtype=dtype,
        )
    )

    hadamard_sdag = (
        inv_sqrt_two
        * jnp.asarray(
            [
                [1, -1j],
                [1, 1j],
            ],
            dtype=dtype,
        )
    )

    paulis = jnp.asarray(
        [
            [
                [1, 0],
                [0, 1],
            ],
            [
                [0, 1],
                [1, 0],
            ],
            [
                [0, -1j],
                [1j, 0],
            ],
            [
                [1, 0],
                [0, -1],
            ],
        ],
        dtype=dtype,
    )

    phase = jnp.asarray(
        [
            [1, 0],
            [0, 1j],
        ],
        dtype=dtype,
    )

    return (
        identity,
        hadamard,
        hadamard_sdag,
        paulis,
        phase,
    )


def _sample_computational_basis(
    key,
    statevectors,
    n: int,
    output_dtype,
):
    probabilities = jnp.real(
        statevectors
        * jnp.conj(statevectors)
    )

    cumulative = jnp.cumsum(
        probabilities,
        axis=-1,
    )

    uniforms = random.uniform(
        key,
        cumulative.shape[:-1],
        dtype=probabilities.dtype,
    )

    basis_samples = jnp.sum(
        cumulative
        < uniforms[..., None],
        axis=-1,
        dtype=jnp.int32,
    )

    basis_samples = jnp.minimum(
        basis_samples,
        (1 << n) - 1,
    )

    shifts = jnp.arange(
        n - 1,
        -1,
        -1,
        dtype=basis_samples.dtype,
    )

    return (
        (
            basis_samples[..., None]
            >> shifts
        )
        & 1
    ).astype(output_dtype)


@jit
def _sample_pauli_statevectors(
    key,
    indices,
    input_statevectors,
):
    n = indices.shape[-1]
    N = indices.shape[-2]

    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (
            input_statevectors.shape[0],
            N,
            input_statevectors.shape[-1],
        ),
    )

    (
        identity,
        hadamard,
        hadamard_sdag,
        _,
        _,
    ) = _single_qubit_gate_tables(
        statevectors.dtype
    )

    measurement_gates = jnp.stack(
        [
            hadamard,
            hadamard_sdag,
            identity,
        ],
        axis=0,
    )

    def body(
        wire,
        statevectors,
    ):
        gates = measurement_gates[
            indices[..., wire]
        ]

        return _apply_single_qubit_gates(
            statevectors,
            gates,
            wire,
            n,
        )

    statevectors = lax.fori_loop(
        0,
        n,
        body,
        statevectors,
        unroll=False,
    )

    return _sample_computational_basis(
        key,
        statevectors,
        n,
        indices.dtype,
    )


@jit
def _sample_sgm_statevectors(
    key,
    indices,
    input_statevectors,
):
    n = indices.shape[-1] - 1
    N = indices.shape[-2]

    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (
            input_statevectors.shape[0],
            N,
            input_statevectors.shape[-1],
        ),
    )

    blocks = indices[..., :n]
    xy_settings = indices[..., n]

    sorted_wires = jnp.argsort(
        -blocks,
        axis=-1,
        stable=True,
    ).astype(jnp.int32)

    sorted_values = jnp.take_along_axis(
        blocks,
        sorted_wires,
        axis=-1,
    )

    def cnot_body(
        i,
        statevectors,
    ):
        position = n - 2 - i

        active_pair = (
            (
                sorted_values[
                    ...,
                    position,
                ]
                == 1
            )
            & (
                sorted_values[
                    ...,
                    position + 1,
                ]
                == 1
            )
        )

        return _apply_dynamic_cnot(
            statevectors,
            sorted_wires[
                ...,
                position,
            ],
            sorted_wires[
                ...,
                position + 1,
            ],
            active_pair,
            n,
        )

    statevectors = lax.fori_loop(
        0,
        max(n - 1, 0),
        cnot_body,
        statevectors,
        unroll=False,
    )

    real_dtype = jnp.real(
        statevectors
    ).dtype

    cosine = jnp.asarray(
        np.cos(np.pi / 4),
        dtype=real_dtype,
    )

    sine = jnp.asarray(
        np.sin(np.pi / 4),
        dtype=real_dtype,
    )

    rx = jnp.asarray(
        [
            [
                cosine,
                -1j * sine,
            ],
            [
                -1j * sine,
                cosine,
            ],
        ],
        dtype=statevectors.dtype,
    )

    ry = jnp.asarray(
        [
            [
                cosine,
                -sine,
            ],
            [
                sine,
                cosine,
            ],
        ],
        dtype=statevectors.dtype,
    )

    rotations = jnp.where(
        (
            xy_settings
            == 0
        )[
            ...,
            None,
            None,
        ],
        rx,
        ry,
    )

    active_rotation = (
        sorted_values[..., 0]
        == 1
    )

    rotation_wires = (
        sorted_wires[..., 0]
    )

    transformed = (
        _apply_dynamic_single_qubit_gates(
            statevectors,
            rotations,
            rotation_wires,
            n,
        )
    )

    statevectors = jnp.where(
        active_rotation[..., None],
        transformed,
        statevectors,
    )

    return _sample_computational_basis(
        key,
        statevectors,
        n,
        indices.dtype,
    )


def _apply_batched_pauli(
    statevectors,
    x,
    z,
    sign_bits,
    basis,
    shifts,
):
    x_masks = jnp.sum(
        jnp.left_shift(x.astype(jnp.int32), shifts),
        axis=-1,
        dtype=jnp.int32,
    )
    z_masks = jnp.sum(
        jnp.left_shift(z.astype(jnp.int32), shifts),
        axis=-1,
        dtype=jnp.int32,
    )
    permutations = jnp.bitwise_xor(
        basis,
        x_masks[..., None],
    )

    y_counts = lax.population_count(
        jnp.bitwise_and(x_masks, z_masks)
    )
    z_parities = jnp.bitwise_and(
        lax.population_count(
            jnp.bitwise_and(
                basis,
                z_masks[..., None],
            )
        ),
        1,
    )
    phase_exponents = jnp.mod(
        (2 * sign_bits + 3 * y_counts)[..., None]
        + 2 * z_parities,
        4,
    )
    phases = jnp.asarray(
        [1, 1j, -1, -1j],
        dtype=statevectors.dtype,
    )[phase_exponents]

    return phases * jnp.take_along_axis(
        statevectors,
        permutations,
        axis=-1,
    )


@jit
def _sample_clifford_projector_statevectors(
    key,
    indices,
    input_statevectors,
):
    """Sample U|psi> by measuring the commuting Udag Z_i U Paulis."""
    n = indices.shape[-2]
    N = indices.shape[1]
    x, z, sign_bits = jax.vmap(
        jax.vmap(computational_basis_measurement_paulis)
    )(indices)

    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (
            input_statevectors.shape[0],
            N,
            input_statevectors.shape[-1],
        ),
    )
    basis = jnp.arange(
        1 << n,
        dtype=jnp.int32,
    )
    shifts = jnp.arange(
        n - 1,
        -1,
        -1,
        dtype=jnp.int32,
    )
    keys = random.split(key, n)
    outcomes = jnp.zeros(
        (
            input_statevectors.shape[0],
            N,
            n,
        ),
        dtype=indices.dtype,
    )

    def body(i, carry):
        statevectors, outcomes = carry
        pauli_statevectors = _apply_batched_pauli(
            statevectors,
            x[..., i, :],
            z[..., i, :],
            sign_bits[..., i],
            basis,
            shifts,
        )
        expectations = jnp.real(
            jnp.sum(
                jnp.conj(statevectors) * pauli_statevectors,
                axis=-1,
            )
        )
        expectations = jnp.clip(
            expectations,
            -1,
            1,
        )
        probability_zero = (1 + expectations) / 2
        outcome = (
            random.uniform(
                keys[i],
                probability_zero.shape,
                dtype=probability_zero.dtype,
            )
            >= probability_zero
        ).astype(indices.dtype)
        eigenvalue = (
            1
            - 2 * outcome.astype(expectations.dtype)
        )
        denominator = jnp.sqrt(
            jnp.maximum(
                2 * (1 + eigenvalue * expectations),
                jnp.finfo(expectations.dtype).tiny,
            )
        )
        statevectors = (
            statevectors
            + eigenvalue[..., None] * pauli_statevectors
        ) / denominator[..., None]
        outcomes = outcomes.at[..., i].set(outcome)

        return statevectors, outcomes

    _, outcomes = lax.fori_loop(
        0,
        n,
        body,
        (statevectors, outcomes),
        unroll=False,
    )

    return outcomes
