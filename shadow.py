"""Classical-shadow protocols and their simulator-independent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache, partial
from typing import Any, Callable

import catalyst
from flax import struct
import jax
from jax import Array, lax, random
from jax import numpy as jnp
import pennylane as qp
from pennylane import qjit

from tools import clifford
from tools.clifford import (
    GHZ_type_state_clifford,
    GHZ_type_state_clifford_rev,
    Tableau,
)
from tools.distributions import binomial, uniform
from tools.estimator import Estimator
from tools.observable import Observable, PauliObservable
from tools.state import GHZType, State
from tools.statevector import (
    _sample_clifford_projector_statevectors,
    _sample_pauli_statevectors,
    _sample_sgm_statevectors,
)
from tools.utils import (
    PartialCircuit,
    build_parallel_entangler_blocks,
    canonical_form,
    create_tableau,
    permutation_to_swaps,
)


class Shadow(ABC, struct.PyTreeNode):
    """Common lifecycle for sampling and estimating classical shadows.

    Concrete protocols provide their settings, circuit, fast statevector
    kernel, tableau transformations, and inverse channel. The public surface
    stays small: construction, sampling, snapshot creation, and estimation.
    """

    estimator: Estimator
    sample_idx_range: Array
    indices: Array
    outcomes: Array
    U: PartialCircuit
    snapshots: Tableau | None
    device: str = struct.field(pytree_node=False, default="lightning.qubit")
    simulator: str = struct.field(pytree_node=False, default="pennylane")
    noise_fun: Callable | None = struct.field(pytree_node=False, default=None)
    noise_keys: Array | None = None

    # Construction and dimensions

    @classmethod
    def init(
        cls,
        key: Array,
        n: int,
        N: int,
        N_state_reps: int = 1,
        sample_idx_range: Array = jnp.array([], dtype=jnp.int32),
        estimator: Estimator = Estimator(),
        device: str = "lightning.qubit",
        simulator: str = "pennylane",
        noise_fun: Callable | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Shadow":
        """Construct an unsampled shadow object and its measurement settings.

        Args:
            key: JAX random key used for settings and optional noise keys.
            n: Number of qubits.
            N: Number of measurements per state repetition.
            N_state_reps: Number of independently prepared states.
            sample_idx_range: Lower and exclusive upper values for default
                setting sampling.
            estimator: Estimator used to reduce weak values.
            device: PennyLane device used by circuit sampling and ground truth.
            simulator: Sampling backend: `pennylane`, `statevector`, or
                `clifford`.
            noise_fun: Optional circuit function applying shot-level noise.
            *args: Protocol-specific setting arguments.
            **kwargs: Protocol-specific setting and circuit arguments.

        Returns:
            A shadow with sampled settings and empty outcomes.
        """

        sample_idx_range = jnp.asarray(sample_idx_range, dtype=jnp.int32)
        setting_key, noise_key = random.split(key)
        setting_keys = random.split(setting_key, N_state_reps)

        def sample_settings(sample_key):
            return cls.sample_indices(
                sample_key,
                (N, n),
                sample_idx_range,
                *args,
                **kwargs,
            )

        indices = jax.vmap(sample_settings)(setting_keys).astype(jnp.int32)
        static_args, dynamic_args = cls.get_U_params(N=N, n=n, **kwargs)
        circuit = PartialCircuit(
            cls.U_fun,
            static_args=dict(static_args),
            dynamic_args=dict(dynamic_args),
        )

        return cls(
            estimator=estimator,
            sample_idx_range=sample_idx_range,
            indices=indices,
            outcomes=jnp.empty((N_state_reps, N, n), dtype=jnp.int32),
            U=circuit,
            snapshots=Tableau.create(n, N, N_state_reps),
            device=device,
            simulator=simulator,
            noise_fun=noise_fun,
            noise_keys=random.split(noise_key, (N_state_reps, N)),
        )

    @property
    def N_state_reps(self) -> int:
        return self.outcomes.shape[-3]

    @property
    def N(self) -> int:
        return self.outcomes.shape[-2]

    @property
    def n(self) -> int:
        return self.outcomes.shape[-1]

    # Sampling

    def sample(self, key: Array, states: State) -> "Shadow":
        """Sample measurement outcomes with the configured backend.

        Args:
            key: JAX random key used for stochastic measurement outcomes.
            states: Batched states matching `N_state_reps`.

        Returns:
            A replacement shadow containing sampled integer outcomes.

        Raises:
            ValueError: If the simulator name is unsupported.
        """

        samplers = {
            "pennylane": self._sample_pennylane,
            "statevector": self._sample_statevector_backend,
            "clifford": self._sample_clifford,
        }
        sampler = samplers.get(self.simulator)
        if sampler is None:
            choices = ", ".join(samplers)
            raise ValueError(
                f"Unknown simulator {self.simulator!r}; expected one of {choices}"
            )
        return self.replace(outcomes=sampler(key, states))

    def _sample_pennylane(self, key: Array, states: State) -> Array:
        del key
        U_treedef = jax.tree_util.tree_structure(self.U)
        state_treedef = jax.tree_util.tree_structure(states)
        circuit = self._sample_circuit(
            self.n,
            U_treedef,
            state_treedef,
            self.device,
            self.noise_fun,
            self.U.circuit_fun,
        )

        def sample_state(indices, state, noise_keys):
            return circuit(indices, self.U, state, noise_keys)[:, 0].astype(
                jnp.int32
            )

        return jax.vmap(sample_state)(self.indices, states, self.noise_keys)

    def _sample_statevector_backend(self, key: Array, states: State) -> Array:
        # States without an explicit vector retain the original circuit fallback.
        if not hasattr(states, "state_dm"):
            return self._sample_pennylane(key, states)
        return self._sample_statevector(states, key)

    def _sample_clifford(self, key: Array, states: State) -> Array:
        tableaus = jax.vmap(
            lambda state: state.clifford(Tableau.create(self.n))
        )(states)
        keys = random.split(key, (self.N_state_reps, self.N))

        def sample_one(sample_key, index, tableau):
            return self.circuit_clifford(index, tableau).sample(sample_key)

        return jax.vmap(
            jax.vmap(sample_one, in_axes=(0, 0, None)),
            in_axes=(0, 0, 0),
        )(keys, self.indices, tableaus)

    # Reconstruction and estimation

    def create_snapshots(self) -> "Shadow":
        """Reconstruct stabilizer snapshots from settings and outcomes.

        Returns:
            A replacement shadow containing one tableau per measurement.
        """

        snapshots = jax.vmap(jax.vmap(self.inverse_circuit_clifford))(
            self.outcomes, self.indices
        )
        return self.replace(snapshots=snapshots)

    def estimate_weak_property(
        self, snapshots: Tableau, observable: Observable
    ) -> Array:
        """Evaluate one observable on a sequence of snapshots.

        Args:
            snapshots: Snapshot tableaus for one state repetition.
            observable: One observable, including its scalar weight.

        Returns:
            One inverse-channel-corrected weak value per snapshot.
        """

        def estimate_one(tableau):
            expectation = tableau.expval(observable.params)
            return (
                self._inverse_channel(self.n, expectation, observable)
                * observable.weight
            )

        return jax.vmap(estimate_one)(snapshots)

    def estimate_weak_properties(
        self,
        observables: Observable,
        batch_size: int | None = None,
    ) -> Array:
        """Evaluate a batch of observables on all snapshots.

        Args:
            observables: Batched observables.
            batch_size: Optional state-axis mapping batch size.

        Returns:
            Weak values with state, observable, and sample axes.
        """

        def estimate_state(snapshots):
            return jax.vmap(
                lambda observable: self.estimate_weak_property(
                    snapshots, observable
                )
            )(observables)

        return lax.map(estimate_state, self.snapshots, batch_size=batch_size)

    def estimate_properties(
        self,
        observables: Observable,
        Ns: Array = jnp.array([0], dtype=jnp.int32),
        batch_size: int | None = None,
    ) -> Array:
        """Reduce weak values at each requested sample count.

        Args:
            observables: Batched observables.
            Ns: Prefix sample counts passed to the estimator.
            batch_size: Optional state-axis mapping batch size.

        Returns:
            Estimates with state, observable, and prefix-count axes.
        """

        weak_values = self.estimate_weak_properties(observables, batch_size)

        def estimate_observable(values):
            return jax.vmap(lambda N: self.estimator(values, N))(Ns)

        def estimate_state(state_values):
            return jax.vmap(estimate_observable)(state_values)

        return lax.map(estimate_state, weak_values, batch_size=batch_size)

    # Ground truth

    def _ground_truth(self, states: State, observable: Observable) -> Array:
        circuit = self._ground_truth_circuit(
            self.n,
            jax.tree_util.tree_structure(states),
            jax.tree_util.tree_structure(observable),
            self.device,
        )
        return jnp.asarray(
            circuit(states, observable) * observable.weight,
            dtype=jnp.float32,
        )

    def ground_truth(self, states: State, observables: Observable) -> Array:
        """Evaluate exact expectation values for states and observables.

        Args:
            states: Batched state representations.
            observables: Batched observables.

        Returns:
            Weighted expectations with state and observable axes.

        Raises:
            NotImplementedError: If the tableau backend receives a state or
                observable type outside its supported GHZ/Pauli combination.
        """

        if self.simulator == "clifford":
            if not isinstance(states, GHZType) or not isinstance(
                observables, PauliObservable
            ):
                raise NotImplementedError(
                    "Clifford ground truth requires GHZType states and "
                    "PauliObservable observables"
                )
            tableaus = jax.vmap(
                lambda state: state.clifford(Tableau.create(self.n))
            )(states)
            values = jax.vmap(
                lambda tableau: jax.vmap(tableau.expval)(observables.params)
            )(tableaus)
            return values.astype(jnp.float32) * observables.weight

        return jax.vmap(
            self._ground_truth,
            in_axes=(None, 0),
            out_axes=1,
        )(states, observables)

    # Cached PennyLane circuits

    @staticmethod
    @lru_cache(None)
    def _sample_circuit(
        n: int,
        U_treedef,
        state_treedef,
        device: str,
        noise_fun,
        circuit_fun,
    ):
        U_axes = U_treedef.unflatten([None] * U_treedef.num_leaves)
        state_axes = state_treedef.unflatten([None] * state_treedef.num_leaves)
        device_instance = qp.device(
            device,
            wires=range(n),
            c_dtype=jnp.complex64,
        )

        if noise_fun is None:

            @qp.set_shots(1)
            @qp.qnode(device_instance)
            def circuit(index, U, state, noise_key):
                del noise_key
                state()
                circuit_fun(index, **U.static_args, **U.dynamic_args)
                return qp.sample()

        else:

            @qp.set_shots(1)
            @qp.qnode(device_instance)
            def circuit(index, U, state, noise_key):
                state()
                noise_fun(key=noise_key, n=n)
                circuit_fun(index, **U.static_args, **U.dynamic_args)
                return qp.sample()

        batched = catalyst.vmap(
            circuit,
            in_axes=(0, U_axes, state_axes, 0),
        )
        return qjit(autograph=True)(batched)

    @staticmethod
    @lru_cache(None)
    def _ground_truth_circuit(n: int, state_treedef, obs_treedef, device: str):
        state_axes = state_treedef.unflatten([0] * state_treedef.num_leaves)
        obs_axes = obs_treedef.unflatten([None] * obs_treedef.num_leaves)

        @qp.qnode(qp.device(device, wires=range(n), c_dtype=jnp.complex64))
        def circuit(state, observable):
            state()
            observable.circuit()
            return qp.expval(observable.op())

        return qjit(autograph=True)(
            catalyst.vmap(circuit, in_axes=(state_axes, obs_axes))
        )

    # Protocol hooks

    @classmethod
    def sample_indices(
        cls,
        key: Array,
        shape: tuple[int, ...],
        sample_idx_range: Array,
        *args,
        **kwargs,
    ) -> Array:
        del cls, args, kwargs
        return random.randint(key, shape, *sample_idx_range, dtype=jnp.int32)

    @staticmethod
    def get_U_params(**kwargs):
        del kwargs
        return {}, {}

    @staticmethod
    @abstractmethod
    def U_fun(index: Array, **kwargs) -> None:
        """Apply the sampled measurement circuit."""

    @abstractmethod
    def _sample_statevector(self, states: State, key: Array) -> Array:
        """Return outcomes from the specialized statevector kernel."""

    @classmethod
    @abstractmethod
    def circuit_clifford(cls, index: Array, tableau: Tableau) -> Tableau:
        """Apply the sampled circuit to a stabilizer tableau."""

    @classmethod
    def inverse_circuit_clifford(
        cls, outcome: Array, index: Array
    ) -> Tableau:
        raise NotImplementedError(f"{cls.__name__} does not construct snapshots")

    def _inverse_channel(
        self, n: int, expectation: Array, observable: Observable
    ) -> Array:
        raise NotImplementedError


class SGMShadow(Shadow):
    """Selective global measurement (SGM) classical shadow.

    Each setting contains a selective binary block and one global X/Y choice.
    Reconstruction uses either the binomial-channel eigenvalues determined by
    `block_probability` or explicit fixed-weight eigenvalue tables.
    """

    block_probability: float = struct.field(pytree_node=False, default=0.5)
    fixed_weight_eigenvalues: tuple | None = struct.field(
        pytree_node=False,
        default=None,
    )

    @classmethod
    def init(
        cls,
        key: Array,
        n: int,
        N: int,
        N_state_reps: int,
        sample_idx_range: Array = jnp.array([0, 2], dtype=jnp.int32),
        estimator: Estimator = Estimator(),
        distribution: Callable = uniform,
        device: str = "lightning.qubit",
        simulator: str = "pennylane",
        block_probability: float | None = None,
        fixed_weight_eigenvalues: tuple | None = None,
        noise_fun: Callable | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> "SGMShadow":
        """Construct an SGM shadow and sample its measurement settings.

        Args:
            key: JAX random key.
            n: Number of qubits.
            N: Number of measurements per state repetition.
            N_state_reps: Number of state repetitions.
            sample_idx_range: Setting range forwarded to the distribution.
            estimator: Weak-value estimator.
            distribution: Callable sampling SGM setting rows.
            device: PennyLane device for circuit operations.
            simulator: Sampling backend.
            block_probability: Bernoulli block probability used by the inverse
                channel. It is inferred for built-in uniform and binomial
                distributions.
            fixed_weight_eigenvalues: Optional non-Z and Z-type eigenvalue
                tables for optimized fixed-weight sampling.
            noise_fun: Optional circuit noise function.
            *args: Additional distribution arguments.
            **kwargs: Additional distribution and circuit arguments.

        Returns:
            An unsampled SGM shadow.

        Raises:
            ValueError: If a custom distribution omits `block_probability` or
                the probability is outside `(0, 1)`.
        """

        if block_probability is None:
            if distribution is uniform or fixed_weight_eigenvalues is not None:
                block_probability = 0.5
            elif (
                isinstance(distribution, partial)
                and distribution.func is binomial
                and distribution.keywords is not None
                and "q" in distribution.keywords
            ):
                block_probability = float(distribution.keywords["q"])
            else:
                raise ValueError(
                    "block_probability is required for a custom SGM distribution"
                )
        if not 0.0 < block_probability < 1.0:
            raise ValueError("block_probability must lie strictly between 0 and 1")

        shadow = super().init(
            key,
            n,
            N,
            N_state_reps,
            sample_idx_range=sample_idx_range,
            estimator=estimator,
            device=device,
            simulator=simulator,
            noise_fun=noise_fun,
            distribution=distribution,
            *args,
            **kwargs,
        )
        return shadow.replace(
            block_probability=block_probability,
            fixed_weight_eigenvalues=fixed_weight_eigenvalues,
        )

    @classmethod
    def sample_indices(
        cls,
        key: Array,
        shape: tuple[int, ...],
        sample_idx_range: Array,
        distribution: Callable,
        *args,
        **kwargs,
    ) -> Array:
        del cls
        N, n = shape
        return jnp.asarray(
            distribution(key, N, n, sample_idx_range, *args, **kwargs),
            dtype=jnp.int32,
        )

    @staticmethod
    def U_fun(index: Array, **kwargs) -> None:
        del kwargs
        n = index.shape[0] - 1
        build_parallel_entangler_blocks(index[:n], n, index[n])

    def _sample_statevector(self, states: State, key: Array) -> Array:
        return _sample_sgm_statevectors(key, self.indices, states.state_dm)

    @classmethod
    def circuit_clifford(cls, index: Array, tableau: Tableau) -> Tableau:
        del cls
        n = tableau.n
        return GHZ_type_state_clifford(index[:n], index[n], tableau)

    @classmethod
    def inverse_circuit_clifford(
        cls, outcome: Array, index: Array
    ) -> Tableau:
        del cls
        n = outcome.shape[-1]
        tableau = Tableau.create(n).MultiPauli(outcome)
        return GHZ_type_state_clifford_rev(index[:n], index[n], tableau)

    def _measurement_eigenvalue(
        self, n: int, observable: PauliObservable
    ) -> Array:
        n_xy = jnp.sum(
            (observable.params == 1) | (observable.params == 2),
            dtype=jnp.int32,
        )
        n_z = jnp.sum(observable.params == 3, dtype=jnp.int32)

        if self.fixed_weight_eigenvalues is not None:
            non_z, z_type = self.fixed_weight_eigenvalues
            return lax.cond(
                observable.is_ZType,
                lambda: jnp.asarray(z_type, dtype=jnp.float32)[n_z],
                lambda: jnp.asarray(non_z, dtype=jnp.float32)[n_xy],
            )

        q = jnp.asarray(self.block_probability, dtype=jnp.float32)
        non_z = 0.5 * q**n_xy * (1.0 - q) ** (n - n_xy)
        z_type = 0.5 * (1.0 + (1.0 - 2.0 * q) ** n_z)
        return lax.cond(
            observable.is_ZType,
            lambda: z_type,
            lambda: non_z,
        )

    def _inverse_channel(
        self,
        n: int,
        expectation: Array,
        observable: Observable,
    ) -> Array:
        if not isinstance(observable, PauliObservable):
            raise NotImplementedError("SGM reconstruction requires Pauli observables")
        return expectation / self._measurement_eigenvalue(n, observable)


class PauliShadow(Shadow):
    """Product-Pauli shadow with direct weak-value reconstruction.

    Each qubit is independently measured in X, Y, or Z. Weak values are
    reconstructed without materializing stabilizer snapshots.
    """

    @classmethod
    def init(
        cls,
        key: Array,
        n: int,
        N: int,
        N_state_reps: int,
        sample_idx_range: Array = jnp.array([0, 3], dtype=jnp.int32),
        estimator: Estimator = Estimator(),
        *args,
        **kwargs,
    ) -> "PauliShadow":
        del sample_idx_range
        return super().init(
            key,
            n,
            N,
            N_state_reps,
            sample_idx_range=jnp.array([0, 3], dtype=jnp.int32),
            estimator=estimator,
            *args,
            **kwargs,
        )

    @staticmethod
    def U_fun(index: Array, **kwargs) -> None:
        del kwargs
        for wire in range(index.shape[0]):
            if index[wire] == 0:
                qp.Hadamard(wire)
            elif index[wire] == 1:
                qp.adjoint(qp.S)(wire)
                qp.Hadamard(wire)

    def _sample_statevector(self, states: State, key: Array) -> Array:
        return _sample_pauli_statevectors(key, self.indices, states.state_dm)

    @classmethod
    def circuit_clifford(cls, index: Array, tableau: Tableau) -> Tableau:
        del cls
        tableau = tableau.MultiSdag(index == 1)
        return tableau.MultiHadamard(index < 2)

    def create_snapshots(self) -> "PauliShadow":
        return self

    def estimate_weak_properties(
        self,
        observables: Observable,
        batch_size: int | None = None,
    ) -> Array:
        if not isinstance(observables, PauliObservable):
            raise NotImplementedError(
                "Pauli reconstruction requires Pauli observables"
            )

        def estimate_state(data):
            outcomes, indices = data
            params = observables.params[:, None, :]
            eigenvalues = (1 - 2 * outcomes)[None, :, :]
            factors = jnp.where(
                params == 0,
                1.0,
                jnp.where(indices[None, :, :] == params - 1, 3 * eigenvalues, 0),
            )
            return jnp.prod(factors, axis=-1) * observables.weight[:, None]

        return lax.map(
            estimate_state,
            (self.outcomes, self.indices),
            batch_size=batch_size,
        )


class CliffordShadow(Shadow):
    """Global-Clifford shadow sampled through canonical symplectic tableaus."""

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
        N, n = shape
        pauli_key, tableau_key = random.split(key)
        paulis = super().sample_indices(
            pauli_key,
            (N, n),
            jnp.array([0, 3], dtype=jnp.int32),
        )
        tableau_data = jax.vmap(create_tableau, in_axes=(0, None))(
            random.split(tableau_key, N), n
        )
        gamma_delta = tableau_data[:4]
        swap_indices = jax.vmap(permutation_to_swaps)(tableau_data[5])
        return jnp.concatenate(
            gamma_delta
            + (tableau_data[4][:, :, None],)
            + (paulis[:, :, None],)
            + (swap_indices,)
            + (tableau_data[5][:, :, None],),
            axis=-1,
        )

    @staticmethod
    def U_fun(index: Array, **kwargs) -> None:
        del kwargs
        n = index.shape[0]
        gamma_delta = [index[:, i * n : (i + 1) * n] for i in range(4)]
        h_and_O = [index[:, 4 * n + i] for i in range(2)]
        canonical_form(
            *gamma_delta,
            *h_and_O,
            swap_indices=index[:, 4 * n + 2 : 4 * n + 4],
        )

    def _sample_statevector(self, states: State, key: Array) -> Array:
        return _sample_clifford_projector_statevectors(
            key,
            self.indices,
            states.state_dm,
        )

    @staticmethod
    def _tableau_parameters(index: Array, n: int):
        gamma_delta = [index[:, i * n : (i + 1) * n] for i in range(4)]
        h_and_O = [index[:, 4 * n + i] for i in range(2)]
        return gamma_delta, h_and_O, index[:, 4 * n + 4]

    @classmethod
    def circuit_clifford(cls, index: Array, tableau: Tableau) -> Tableau:
        gamma_delta, h_and_O, permutation = cls._tableau_parameters(
            index, tableau.n
        )
        return clifford.canonical_form(
            tableau,
            *gamma_delta,
            *h_and_O,
            permutation,
        )

    @classmethod
    def inverse_circuit_clifford(
        cls, outcome: Array, index: Array
    ) -> Tableau:
        tableau = Tableau.create(outcome.shape[-1]).MultiPauli(outcome)
        gamma_delta, h_and_O, permutation = cls._tableau_parameters(
            index, tableau.n
        )
        return clifford.canonical_form_rev(
            tableau,
            *gamma_delta,
            *h_and_O,
            permutation,
        )

    def _inverse_channel(
        self,
        n: int,
        expectation: Array,
        observable: Observable,
    ) -> Array:
        if not isinstance(observable, PauliObservable):
            raise NotImplementedError(
                "Clifford reconstruction requires Pauli observables"
            )
        return (
            jnp.float32(2**n) + 1
        ) * jnp.asarray(expectation, dtype=jnp.float32) - jnp.real(
            jnp.asarray(observable.trace(), dtype=jnp.complex64)
        )
