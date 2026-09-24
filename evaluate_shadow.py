"""Execution and persistence for shadow-scaling experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import os
import time
from typing import Any

import jax
from jax import Array, jit, random, tree
from jax import numpy as jnp
import numpy as np

from shadow import Shadow
from tools.batching import BatchedRunner
from tools.estimator import Estimator
from tools.observable import Observable
from tools.state import State


class ShadowScalingExperiment:
    """Evaluate one shadow protocol over qubit and sample-count axes.

    Args:
        shadow_cls: Shadow protocol to construct for every qubit count.
        state_cls: Random-state factory used for every repetition.
        shadow_args: Function returning protocol keyword arguments for a qubit
            count.
        N_list: Requested prefix sample counts.
        n_list: Qubit counts to evaluate.
        obs_lists: One observable collection per qubit count.
        gate_indices: Function returning the measurement-setting index range.
        estimator: Function returning an estimator for a qubit and sample
            count.
        N_state_reps: Number of independently sampled states.
        key: JAX random key controlling states, settings, and outcomes.
        verbose: Whether to print cache and timing progress.
        bs_state: State-repetition batch size.
        path_save: Result prefix without the `.npy` extension, or `None` to
            disable persistence.
        batch_size_obs: Observable batch size. `None` selects all observables.
        batch_size_N: Sample batch size. `None` selects all samples.
        experiment_name: Human-readable label used in logs and plots.

    Raises:
        ValueError: If the number of observable collections differs from the
            number of qubit counts.

    Notes:
        Batched execution requires the maximum sample count, repetition count,
        and observable count to be divisible by their respective batch sizes.
    """

    def __init__(
        self,
        shadow_cls: type[Shadow],
        state_cls: type[State],
        shadow_args: Callable[[int], dict[str, Any]] = lambda n: {},
        N_list: Sequence[int] = (),
        n_list: Sequence[int] = (),
        obs_lists: Sequence[Observable] = (),
        gate_indices: Callable[[int], Array] = lambda n: jnp.array(
            [0, 2], dtype=jnp.int32
        ),
        estimator: Callable[[int, int], Estimator] = lambda n, N: Estimator(),
        N_state_reps: int = 1,
        key: Array = random.PRNGKey(12345),
        verbose: bool = False,
        bs_state: int = 1,
        path_save: str | None = None,
        batch_size_obs: int | None = None,
        batch_size_N: int | None = None,
        experiment_name: str | None = None,
    ) -> None:
        if len(obs_lists) != len(n_list):
            raise ValueError("Need one observable collection per qubit count")

        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.Ns = jnp.asarray(N_list, dtype=jnp.int32)
        self.N = int(jnp.max(self.Ns))
        self.ns = list(n_list)
        self.N_state_reps = N_state_reps
        self.observables_list = list(obs_lists)
        self.gate_indices = gate_indices
        self.estimator = estimator
        self.verbose = verbose
        self.bs_state = bs_state
        self.bs_N = batch_size_N if batch_size_N is not None else self.N
        self.bs_obs = (
            batch_size_obs
            if batch_size_obs is not None
            else self.observables_list[0].params.shape[0]
        )
        self.use_batched = batch_size_N is not None or batch_size_obs is not None
        self.path_save = path_save
        self.experiment_name = experiment_name

        result_shape = (
            len(self.Ns),
            len(self.ns),
            self.observables_list[0].params.shape[0],
        )
        self.preds = jnp.zeros(result_shape, dtype=jnp.float32)
        self.gt = jnp.zeros_like(self.preds)

        if self.use_batched:
            self.states = []
            self.shadows = []
        else:
            self._initialize(key)
        self.key = random.split(key)[1]

    def _initialize(self, key) -> None:
        state_keys, shadow_keys = random.split(key, (2, len(self.ns)))
        self.states = [
            self.state_cls.init_random(
                state_keys[index], self.N_state_reps, n
            )
            for index, n in enumerate(self.ns)
        ]
        self.shadows = [
            self.shadow_cls.init(
                shadow_keys[index],
                n,
                self.N,
                self.N_state_reps,
                self.gate_indices(n),
                self.estimator(n, self.N),
                **self.shadow_args(n),
            )
            for index, n in enumerate(self.ns)
        ]

    def run(self) -> None:
        """Load cached results or execute and persist the experiment."""

        label = self.experiment_name or self.shadow_cls.__name__
        if self.path_save is not None and os.path.isfile(self.path_save + ".npy"):
            if self.verbose:
                print(f"[{label}] loading {self.path_save}.npy")
            self.load_results(self.path_save + ".npy")
            return

        if self.path_save is not None and self.verbose:
            print(f"[{label}] results will be saved to {self.path_save}.npy")

        start = time.perf_counter()
        if self.use_batched:
            results = self.run_batched(self.observables_list)
            predictions, ground_truth = zip(*results)
            self.preds = jnp.stack(predictions, axis=0)
            self.gt = jnp.stack(ground_truth, axis=0)
        else:
            self._sample_shadows()
            self.preds, self.gt = self.estimate_full(self.observables_list)
        jax.block_until_ready((self.preds, self.gt))

        if self.verbose:
            elapsed = time.perf_counter() - start
            n_observables = self.observables_list[0].params.shape[0]
            print(
                f"[{label}] estimated {n_observables} observables from "
                f"{self.N_state_reps} states and {self.N} samples in "
                f"{elapsed:.3f} s"
            )

        if self.path_save is not None:
            self.save_results(self.path_save)

    def _sample_shadows(self) -> None:
        is_leaf = lambda value: isinstance(
            value, (self.shadow_cls, self.state_cls, Observable)
        )
        self.key, sample_key = random.split(self.key)
        keys = list(random.split(sample_key, len(self.ns)))

        @jit
        def sample_all(shadows, states):
            return tree.map(
                lambda shadow, state, key: shadow.sample(key, state),
                shadows,
                states,
                keys,
                is_leaf=is_leaf,
            )

        self.shadows = sample_all(self.shadows, self.states)
        jax.block_until_ready(self.shadows)

    def run_batched(
        self, observables_list: Sequence[Observable]
    ) -> list[tuple[Array, Array]]:
        """Run the streaming batch pipeline for every qubit count.

        Args:
            observables_list: One equally sized observable collection per
                qubit count.

        Returns:
            Prediction and broadcast ground-truth tensors for each qubit count.

        Raises:
            ValueError: If the collection count is wrong, observable counts
                differ, or a batch divisibility invariant is violated.
        """

        if not observables_list or len(observables_list) != len(self.ns):
            raise ValueError("Need one observable batch for each qubit count")
        counts = [observables.params.shape[0] for observables in observables_list]
        if len(set(counts)) != 1:
            raise ValueError(
                "Results are stacked over n, so every qubit count requires "
                "the same number of observables"
            )

        label = self.experiment_name or self.shadow_cls.__name__
        print(
            f"[{label}] batch sizes: states={self.bs_state}, "
            f"observables={self.bs_obs}, samples={self.bs_N}"
        )

        def prepare(observables, n):
            runner = BatchedRunner(
                Ns=self.Ns,
                sample_idx_range=self.gate_indices(n),
                estimator=self.estimator(n, self.N),
                sample_estimator=self.estimator(n, self.bs_N),
                ground_truth_estimator=self.estimator(n, 1),
                n=int(n),
                N=self.N,
                repetitions=int(self.N_state_reps),
                bs_state=int(self.bs_state),
                bs_obs=int(self.bs_obs),
                bs_N=int(self.bs_N),
                state_cls=self.state_cls,
                shadow_cls=self.shadow_cls,
                shadow_kwargs=tuple(sorted(self.shadow_args(n).items())),
                verbose=bool(self.verbose),
            )
            runner.validate(observables)
            return runner

        is_leaf = lambda value: isinstance(value, Observable)
        runners = tree.map(prepare, observables_list, self.ns, is_leaf=is_leaf)
        self.key, run_key = random.split(self.key)
        run_keys = list(random.split(run_key, len(self.ns)))

        def execute(observables, runner, key):
            result = runner.run(key, observables)
            ground_truth = jnp.broadcast_to(
                result.ground_truth[..., None], result.predictions.shape
            )
            return result.predictions, ground_truth

        return tree.map(
            execute,
            observables_list,
            runners,
            run_keys,
            is_leaf=is_leaf,
        )

    def estimate_full(
        self, observables_list: Sequence[Observable]
    ) -> tuple[Array, Array]:
        """Estimate observables from fully materialized shadow samples.

        Args:
            observables_list: One observable collection per qubit count.

        Returns:
            Prediction and ground-truth tensors with a shared shape.
        """

        is_leaf = lambda value: isinstance(
            value, (self.shadow_cls, self.state_cls, Observable)
        )

        @partial(jit, static_argnums=(4,))
        def compute_results(shadows, states, observables, Ns, batch_size):
            shadows = tree.map(
                lambda shadow: shadow.create_snapshots(),
                shadows,
                is_leaf=is_leaf,
            )
            predictions = jnp.asarray(
                tree.map(
                    lambda shadow, obs: shadow.estimate_properties(
                        obs, Ns, batch_size
                    ),
                    shadows,
                    observables,
                    is_leaf=is_leaf,
                ),
                dtype=jnp.float32,
            )
            ground_truth = jnp.asarray(
                tree.map(
                    lambda shadow, state, obs: shadow.ground_truth(state, obs),
                    shadows,
                    states,
                    observables,
                    is_leaf=is_leaf,
                ),
                dtype=jnp.float32,
            )
            ground_truth = jnp.repeat(
                ground_truth[..., None], Ns.shape[0], axis=-1
            )
            return predictions, ground_truth

        return compute_results(
            self.shadows,
            self.states,
            observables_list,
            self.Ns,
            self.bs_state,
        )

    def save_results(self, path: str) -> None:
        """Save ground truth and predictions under a result prefix.

        Args:
            path: Output prefix. NumPy appends `.npy` when needed.
        """

        np.save(path, np.stack([self.gt, self.preds]))

    def load_results(self, path: str) -> None:
        """Load ground truth and predictions from a result array.

        Args:
            path: Path to a `.npy` file produced by `save_results`.

        Notes:
            Cache files contain no metadata and are assumed to match the
            current experiment axes and observables.
        """

        ground_truth, predictions = np.load(path)
        self.gt = jnp.asarray(ground_truth, dtype=jnp.float32)
        self.preds = jnp.asarray(predictions, dtype=jnp.float32)
