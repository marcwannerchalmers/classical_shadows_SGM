"""Pure JAX orchestration for one qubit count and a streaming estimator.

The Python experiment object handles file I/O and its own PRNG key. This
immutable runner contains only configuration and JAX-compatible values.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax import Array, lax, random
from jax import numpy as jnp
from flax import struct
import numpy as np


@struct.dataclass
class BatchKeys:
    state: Array
    shadow: Array
    sample: Array


@struct.dataclass
class BatchResult:
    predictions: Array
    ground_truth: Array


@struct.dataclass
class BatchedRunner:
    Ns: Array
    sample_idx_range: Array
    estimator: Any
    sample_estimator: Any
    ground_truth_estimator: Any

    n: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    repetitions: int = struct.field(pytree_node=False)
    bs_state: int = struct.field(pytree_node=False)
    bs_obs: int = struct.field(pytree_node=False)
    bs_N: int = struct.field(pytree_node=False)
    state_cls: type = struct.field(pytree_node=False)
    shadow_cls: type = struct.field(pytree_node=False)
    shadow_kwargs: tuple[tuple[str, Any], ...] = struct.field(
        pytree_node=False, default=()
    )
    verbose: bool = struct.field(pytree_node=False, default=False)

    def validate(self, observables: Any) -> None:
        """Host-side checks; call before entering run()."""
        for name in ("n", "N", "repetitions", "bs_state", "bs_obs", "bs_N"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer")

        if self.N % self.bs_N:
            raise ValueError("N must be divisible by bs_N")
        if self.repetitions % self.bs_state:
            raise ValueError("repetitions must be divisible by bs_state")
        if observables.params.ndim < 2 or observables.params.shape[0] == 0:
            raise ValueError("observables.params must have a nonempty batch axis")
        if observables.params.shape[0] % self.bs_obs:
            raise ValueError("The observable count must be divisible by bs_obs")
        if not callable(getattr(observables, "replace", None)):
            raise TypeError("Observables must support Flax-style replace()")

        Ns = np.asarray(self.Ns)
        if Ns.ndim != 1 or Ns.size == 0:
            raise ValueError("Ns must be a nonempty one-dimensional array")
        if not np.issubdtype(Ns.dtype, np.integer):
            raise TypeError("Ns must have an integer dtype")
        if np.any(Ns <= 0) or np.any(Ns > self.N):
            raise ValueError("Every requested sample count must satisfy 1 <= N_i <= N")
        self.estimator.validate_Ns(self.Ns)

        # These kwargs were static arguments in the original implementation.
        reserved = {"key", "n", "N", "N_state_reps", "sample_idx_range", "estimator"}
        if reserved.intersection(dict(self.shadow_kwargs)):
            raise ValueError("shadow_kwargs cannot override runner-owned arguments")
        try:
            hash(self.shadow_kwargs)
        except TypeError as exc:
            raise TypeError(
                "shadow_kwargs must contain immutable, hashable configuration. "
                "Runtime array parameters should be separate PyTree data fields, "
                "not static metadata."
            ) from exc

    def _make_keys(self, key: Array) -> BatchKeys:
        """Preserve the original splitting scheme for a fixed batching layout."""
        key_shadow, key_state, key_sample = random.split(key, 3)
        state_batches = self.repetitions // self.bs_state
        sample_batches = self.N // self.bs_N
        return BatchKeys(
            state=random.split(key_state, state_batches),
            shadow=random.split(key_shadow, (state_batches, sample_batches)),
            sample=random.split(key_sample, (state_batches, sample_batches)),
        )

    def _make_shadow(self, key: Array, count: int, estimator: Any) -> Any:
        kwargs = {
            **dict(self.shadow_kwargs),
        }
        return self.shadow_cls.init(
            key=key,
            n=self.n,
            N=count,
            N_state_reps=self.bs_state,
            sample_idx_range=self.sample_idx_range,
            estimator=estimator,
            **kwargs,
        )

    def _slice_observables(self, observables: Any, index: Array) -> Any:
        """Slice the current PauliObservable representation without losing metadata.

        If a future observable has additional per-observable array fields,
        those must be sliced here too (or in its own batch-slicing method).
        """
        params = lax.dynamic_slice_in_dim(
            observables.params, index * self.bs_obs, self.bs_obs, axis=0
        )
        weight = lax.dynamic_slice_in_dim(
            observables.weight, index * self.bs_obs, self.bs_obs, axis=0
        )
        return observables.replace(params=params, weight=weight)

    def _map_observables(
        self, observables: Any, evaluate: Callable[[Any], Array]
    ) -> Array:
        """Map over observable batches, returning (state, observable, ...)."""
        n_obs = observables.params.shape[0]

        def evaluate_batch(index):
            values = jnp.asarray(
                evaluate(self._slice_observables(observables, index)),
                dtype=jnp.float32,
            )
            if values.shape[:2] != (self.bs_state, self.bs_obs):
                raise ValueError(
                    "Observable evaluation must return leading axes "
                    f"(state, observable) = {(self.bs_state, self.bs_obs)}; "
                    f"got {values.shape}"
                )
            return values

        blocks = lax.map(
            evaluate_batch,
            jnp.arange(n_obs // self.bs_obs, dtype=jnp.int32),
        )
        return jnp.swapaxes(blocks, 0, 1).reshape(
            (self.bs_state, n_obs) + blocks.shape[3:]
        )

    def _ground_truth(self, state: Any, observables: Any, key: Array) -> Array:
        shadow = self._make_shadow(key, 1, self.ground_truth_estimator)
        return self._map_observables(
            observables, lambda obs: shadow.ground_truth(state, obs)
        )

    def _sample_step(
        self,
        index: Array,
        estimator_state: Any,
        *,
        keys: BatchKeys,
        state: Any,
        observables: Any,
    ) -> Any:
        shadow = self._make_shadow(
            keys.shadow[index], self.bs_N, self.sample_estimator
        )
        shadow = shadow.sample(keys.sample[index], state)
        shadow = shadow.create_snapshots()
        weak_batch = self._map_observables(
            observables, shadow.estimate_weak_properties
        )
        return self.estimator.update(estimator_state, weak_batch, self.Ns)

    def _state_batch(self, keys: BatchKeys, observables: Any) -> BatchResult:
        state = self.state_cls.init_random(
            key=keys.state, N_state=self.bs_state, n=self.n
        )
        ground_truth = self._ground_truth(state, observables, keys.shadow[0])
        estimator_state = self.estimator.init_state(
            (self.bs_state, observables.params.shape[0]), self.Ns
        )

        def step(index, carry):
            return self._sample_step(
                index, carry, keys=keys, state=state, observables=observables
            )

        sample_batches = self.N // self.bs_N
        if self.verbose:
            from jax_tqdm import loop_tqdm

            step = loop_tqdm(
                sample_batches,
                print_rate=max(1, sample_batches // 20),
                desc="Sample batches",
            )(step)

        estimator_state = lax.fori_loop(0, sample_batches, step, estimator_state)
        predictions = jnp.asarray(
            self.estimator.finalize(estimator_state, self.Ns),
            dtype=jnp.float32,
        )
        expected = (self.bs_state, observables.params.shape[0], self.Ns.shape[0])
        if predictions.shape != expected:
            raise ValueError(
                f"Estimator.finalize returned {predictions.shape}, "
                f"expected {expected}"
            )
        return BatchResult(predictions, ground_truth)

    @jax.jit
    def run(self, key: Array, observables: Any) -> BatchResult:
        keys = self._make_keys(key)
        blocks = lax.map(lambda group: self._state_batch(group, observables), keys)
        return jax.tree.map(
            lambda value: value.reshape((self.repetitions,) + value.shape[2:]),
            blocks,
        )
