"""Sampling distributions for selective global measurements."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
import math
from typing import Any

from jax import Array, random
from jax import numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar


def uniform(
    key: Array,
    N: int,
    n: int,
    sample_idx_range: Sequence[int],
) -> Array:
    """Sample uniform selective blocks and X/Y settings.

    Args:
        key: JAX random key.
        N: Number of settings to sample.
        n: Number of qubits in each selective block.
        sample_idx_range: Lower and exclusive upper block values.

    Returns:
        Integer settings with shape `(N, n + 1)`; the final column is the X/Y
        setting.
    """

    block_key, setting_key = random.split(key)
    blocks = random.randint(
        block_key,
        (N, n),
        *sample_idx_range,
        dtype=jnp.int32,
    )
    settings = random.randint(
        setting_key,
        (N,),
        0,
        2,
        dtype=jnp.int32,
    )
    return jnp.concatenate([blocks, settings[:, None]], axis=1)


def binomial(
    key: Array,
    N: int,
    n: int,
    sample_idx_range: Sequence[int],
    q: float,
) -> Array:
    """Sample Bernoulli block masks and one X/Y setting per shot.

    Args:
        key: JAX random key.
        N: Number of settings to sample.
        n: Number of qubits in each selective block.
        sample_idx_range: Unused compatibility argument.
        q: Probability that each block entry is active.

    Returns:
        Integer settings with shape `(N, n + 1)`.
    """

    del sample_idx_range
    block_key, setting_key = random.split(key)
    blocks = random.bernoulli(
        block_key,
        jnp.asarray(q, dtype=jnp.float32),
        (N, n),
    )
    settings = random.bernoulli(setting_key, shape=(N,))
    return jnp.concatenate([blocks, settings[:, None]], axis=1).astype(jnp.int32)


def fixed_weight(
    key: Array,
    N: int,
    n: int,
    sample_idx_range: Sequence[int],
    probabilities: Sequence[float],
) -> Array:
    """Sample a block weight, then a uniform block of that weight.

    Args:
        key: JAX random key.
        N: Number of settings to sample.
        n: Number of qubits in each selective block.
        sample_idx_range: Unused compatibility argument.
        probabilities: Probability mass for block weights `0` through `n`.

    Returns:
        Integer settings with shape `(N, n + 1)`.
    """

    del sample_idx_range
    weight_key, priority_key, setting_key = random.split(key, 3)
    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    weights = random.categorical(
        weight_key,
        jnp.log(probabilities),
        shape=(N,),
    ).astype(jnp.int32)
    priorities = random.uniform(priority_key, (N, n), dtype=jnp.float32)
    ranks = jnp.argsort(jnp.argsort(priorities, axis=1), axis=1)
    blocks = ranks < weights[:, None]
    settings = random.bernoulli(setting_key, shape=(N,))
    return jnp.concatenate([blocks, settings[:, None]], axis=1).astype(jnp.int32)


def _optimized_binomial_q(n: int, observables: Any) -> float:
    """Minimize the exact average inverse-channel eigenvalue."""

    params = np.asarray(observables.params)
    n_xy = np.count_nonzero((params == 1) | (params == 2), axis=1)
    n_z = np.count_nonzero(params == 3, axis=1)
    non_z = n_xy > 0

    def risk(q: float) -> float:
        non_z_eigenvalues = 0.5 * q ** n_xy[non_z] * (1 - q) ** (
            n - n_xy[non_z]
        )
        z_eigenvalues = 0.5 * (1 + (1 - 2 * q) ** n_z[~non_z])
        return float(
            np.sum(1 / non_z_eigenvalues) + np.sum(1 / z_eigenvalues)
        )

    result = minimize_scalar(
        risk,
        bounds=(1e-6, 1 - 1e-6),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Could not optimize the binomial q: {result.message}")
    return float(result.x)


def _fixed_weight_probabilities(n: int, observables: Any) -> tuple[float, ...]:
    params = np.asarray(observables.params)
    supports = np.count_nonzero((params == 1) | (params == 2), axis=1)
    probabilities = np.zeros(n + 1, dtype=np.float64)
    for support in np.unique(supports[supports > 0]):
        count = np.count_nonzero(supports == support)
        probabilities[support] = np.sqrt(count * math.comb(n, int(support)))
    if probabilities.sum() == 0:
        raise ValueError("fixed_weight requires at least one non-Z observable")
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in probabilities)


def _fixed_weight_eigenvalues(
    n: int,
    probabilities: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    non_z = np.zeros(n + 1, dtype=np.float64)
    z_type = np.zeros(n + 1, dtype=np.float64)
    for support in range(1, n + 1):
        non_z[support] = probabilities[support] / (2 * math.comb(n, support))
    for z_weight in range(n + 1):
        visibility = 0.0
        for weight, probability in enumerate(probabilities):
            even_probability = sum(
                math.comb(z_weight, overlap)
                * math.comb(n - z_weight, weight - overlap)
                for overlap in range(0, z_weight + 1, 2)
                if 0 <= weight - overlap <= n - z_weight
            ) / math.comb(n, weight)
            visibility += probability * even_probability
        z_type[z_weight] = visibility
    return tuple(non_z), tuple(z_type)


def build_sgm_distribution(
    spec: Mapping[str, Any],
    n: int,
    observables: Any,
) -> tuple[
    Callable[..., Array],
    tuple[tuple[float, ...], tuple[float, ...]] | None,
]:
    """Build an SGM sampler and optional fixed-weight eigenvalue tables.

    Args:
        spec: Distribution mapping with a `name` and its parameters.
        n: Number of qubits.
        observables: Observable batch used by optimized distributions.

    Returns:
        The setting sampler and, for fixed-weight sampling, non-Z and Z-type
        channel eigenvalues indexed by support weight.

    Raises:
        ValueError: If the distribution is unknown, its optimization fails,
            required observables are absent, or a binomial probability lies
            outside `(0, 1)`.
    """

    name = spec.get("name", "uniform")
    if name == "uniform":
        return uniform, None
    if name == "fixed_weight":
        if spec.get("weights", "rdm_optim") != "rdm_optim":
            raise ValueError("fixed_weight currently supports weights: rdm_optim")
        probabilities = _fixed_weight_probabilities(n, observables)
        return (
            partial(fixed_weight, probabilities=probabilities),
            _fixed_weight_eigenvalues(n, probabilities),
        )
    if name != "binomial":
        raise ValueError(f"Unknown distribution {name!r}")

    q_spec = spec.get("q")
    if q_spec == "rdm_optim":
        q = _optimized_binomial_q(n, observables)
    elif q_spec == "mean_xy_support":
        xy_support = (observables.params == 1) | (observables.params == 2)
        q = float(jnp.mean(jnp.sum(xy_support, axis=1)) / n)
        q = min(max(q, 1 / (2 * n)), 1 - 1 / (2 * n))
    else:
        q = float(q_spec)
    if not 0 < q < 1:
        raise ValueError(f"Binomial q must lie strictly between zero and one, got {q}")
    return partial(binomial, q=q), None
