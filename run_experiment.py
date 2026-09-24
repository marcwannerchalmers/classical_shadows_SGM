"""Hydra entry point for declarative shadow-scaling experiments."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from hydra import main as hydra_main
from jax import Array, random
from omegaconf import DictConfig, OmegaConf

from evaluate_shadow import ShadowScalingExperiment
from experiment import Experiments
from shadow import CliffordShadow, PauliShadow, SGMShadow
from tools.distributions import build_sgm_distribution
from tools.majorana import (
    FermionicGaussianShadow,
    MajoranaState,
    majorana_observables_list,
)
from tools.noise import depolarizing_noise
from tools.observable import PauliObservable
from tools.state import GHZType, GHZ_Klocal as _GHZ_Klocal, HRState, State


SHADOW_CLASSES = {
    cls.__name__: cls
    for cls in (SGMShadow, PauliShadow, CliffordShadow, FermionicGaussianShadow)
}
STATE_CLASSES = {cls.__name__: cls for cls in (HRState, GHZType, MajoranaState)}
PAULI_INDICES = {"I": 0, "X": 1, "Y": 2, "Z": 3}
PLOT_METHODS = {
    "plot_avg_var",
    "plot_avg_var_zero_segments",
    "plot_sample_complexity",
}


@dataclass
class ExperimentPlan:
    """Parsed experiments and their shared plot configuration.

    Attributes:
        experiments: Experiments in deterministic configuration-expansion
            order.
        plot: Resolved options passed to the plotting layer.
    """

    experiments: list[ShadowScalingExperiment]
    plot: dict[str, Any]


def _load_config(cfg: DictConfig | Mapping[str, Any] | str | Path) -> DictConfig:
    if isinstance(cfg, (str, Path)):
        return OmegaConf.load(cfg)
    if isinstance(cfg, DictConfig):
        # Hydra enables struct mode on the composed config.  Recreating it keeps
        # interpolation intact while allowing experiment entries to override or
        # add their own fields during the merge below.
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    return OmegaConf.create(cfg)


def _integer_sequence(spec: Any) -> list[int]:
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        return [int(value) for value in spec]
    if not isinstance(spec, Mapping):
        return [int(spec)]
    if "values" in spec:
        return _integer_sequence(spec["values"])
    if "range" in spec:
        values = spec["range"]
        return list(
            range(
                int(values.get("start", 0)),
                int(values["stop"]),
                int(values.get("step", 1)),
            )
        )
    if "powers" in spec:
        values = spec["powers"]
        base = int(values.get("base", 2))
        return [
            base**power
            for power in range(
                int(values["start"]),
                int(values["stop"]),
                int(values.get("step", 1)),
            )
        ]
    raise ValueError(f"Expected values, range, or powers, got: {spec}")


def _resolved_section(config: DictConfig, name: str, n: int | None = None) -> dict:
    local = OmegaConf.merge(config, {"n": n}) if n is not None else config
    section = local.get(name, {})
    if OmegaConf.is_config(section):
        return dict(OmegaConf.to_container(section, resolve=True))
    return dict(section)


def _pauli_indices(values: Sequence[str | int]) -> list[int]:
    try:
        return [
            PAULI_INDICES[value.upper()] if isinstance(value, str) else int(value)
            for value in values
        ]
    except KeyError as exc:
        raise ValueError(f"Unknown Pauli label {exc.args[0]!r}") from exc


def _observables(
    config: DictConfig, ns: list[int], key: Array
) -> list[PauliObservable]:
    factory = config.get("observables", {}).get("init", "random_pauli")
    if factory in {"random", "random_pauli"}:
        result = []
        for obs_key, n in zip(random.split(key, len(ns)), ns):
            spec = _resolved_section(config, "observables", n)
            result.append(
                PauliObservable.init_random(
                    obs_key,
                    sample_indices=_pauli_indices(
                        spec.get("sample_indices", ["I", "X", "Y", "Z"])
                    ),
                    n=n,
                    N=int(spec["count"]),
                    k_local=int(spec.get("k_local", 0)),
                    padding_indices=_pauli_indices(
                        spec.get("padding_indices", ["I"])
                    ),
                )
            )
        return result

    if factory in {"majorana", "majorana_rdm"}:
        spec = _resolved_section(config, "observables", ns[0])
        count = int(spec.get("count", -1))
        subset = spec.get("subset", "random") if count != -1 else "all"
        return majorana_observables_list(
            ns, int(spec["order"]), subset, count, key
        )

    if factory in {"file", "npy"} or str(factory).endswith(".npy"):
        spec = _resolved_section(config, "observables", ns[0])
        params = np.load(str(spec.get("path", factory)))
        if params.ndim == 2 and len(ns) == 1:
            params = params[None, ...]
        if params.ndim != 3 or params.shape[0] != len(ns):
            raise ValueError(
                "Observable files must have shape (len(ns), N_obs, n), "
                "or (N_obs, n) for one n"
            )
        return [
            PauliObservable.init(jnp.asarray(values, dtype=jnp.int32))
            for values in params
        ]

    raise ValueError(f"Unknown observable factory {factory!r}")


def _state_class(spec: Mapping[str, Any]):
    name = spec.get("class", "HRState")
    if name == "GHZ_Klocal":
        return _GHZ_Klocal(int(spec["locality"]))
    try:
        return STATE_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown state class {name!r}") from exc


def _shadow_arguments(
    config: DictConfig,
    ns: list[int],
    observables: list[PauliObservable],
):
    arguments_by_n = {}
    for n, observable in zip(ns, observables):
        parsed = {}
        for name, value in _resolved_section(config, "shadow_args", n).items():
            if name == "distribution":
                distribution, eigenvalues = build_sgm_distribution(
                    value, n, observable
                )
                parsed[name] = distribution
                if eigenvalues is not None:
                    parsed["fixed_weight_eigenvalues"] = eigenvalues
            elif name in {"noise", "noise_fun"}:
                if value is None or (
                    isinstance(value, str) and value.lower() == "none"
                ):
                    continue
                if not isinstance(value, Mapping):
                    raise ValueError("noise must be null or a mapping")
                if value.get("name") != "depolarizing":
                    raise ValueError(f"Unknown noise model {value.get('name')!r}")
                parsed["noise_fun"] = partial(
                    depolarizing_noise, p=float(value["p"])
                )
            else:
                parsed[name] = value
        arguments_by_n[n] = parsed
    return arguments_by_n.__getitem__


def _expanded_configs(config: DictConfig):
    """Yield one fully merged config for each experiment/sweep value."""

    for experiment in config["experiments"]:
        merged = OmegaConf.merge(config, experiment)
        sweep = merged.get("for_each")
        if sweep is None:
            yield merged
            continue
        variable = sweep["variable"]
        for value in _integer_sequence(sweep):
            yield OmegaConf.merge(merged, {variable: value})


def _experiment_axes(config: DictConfig) -> tuple[list[int], list[int]]:
    qubits = config.get("qubits", config.get("n"))
    samples = config.get("samples", config.get("N"))
    if qubits is None or samples is None:
        raise ValueError("Each experiment requires qubits and samples")
    return _integer_sequence(qubits), _integer_sequence(samples)


def _build_experiment(
    config: DictConfig,
    observable_key: Array,
    experiment_key: Array,
    repetitions: int,
) -> ShadowScalingExperiment:
    ns, Ns = _experiment_axes(config)
    observables = _observables(config, ns, observable_key)
    state = _resolved_section(config, "state")
    batch = _resolved_section(config, "batch")
    shadow_name = config["shadow_class"]
    try:
        shadow_cls = SHADOW_CLASSES[shadow_name]
    except KeyError as exc:
        raise ValueError(f"Unknown shadow class {shadow_name!r}") from exc

    path = config.get("path_prefix")
    return ShadowScalingExperiment(
        shadow_cls,
        _state_class(state),
        shadow_args=_shadow_arguments(config, ns, observables),
        N_list=Ns,
        n_list=ns,
        obs_lists=observables,
        N_state_reps=repetitions,
        verbose=bool(config.get("verbose", False)),
        key=experiment_key,
        path_save=None if path is None else str(path),
        bs_state=int(batch.get("states", 1)),
        batch_size_obs=batch.get("observables"),
        batch_size_N=batch.get("samples"),
        experiment_name=str(config.get("experiment_name", config.get("id"))),
    )


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_experiment_config(
    cfg: DictConfig | Mapping[str, Any] | str | Path,
) -> ExperimentPlan:
    """Parse an experiment configuration without executing it.

    Args:
        cfg: YAML path, OmegaConf configuration, or equivalent mapping.

    Returns:
        The expanded experiment objects and resolved plot options.

    Raises:
        ValueError: If the configuration names an unsupported state, shadow,
            observable factory, distribution, noise model, or axis format.

    Notes:
        Sweep entries are expanded in configuration order. Random-key splitting
        is therefore sensitive to experiment ordering.
    """

    config = _load_config(cfg)
    key = random.PRNGKey(int(config.get("seed", 12345)))
    key, _ = random.split(key)  # Preserve the reference PRNG split order.
    experiments = []
    repetitions = int(config["repetitions"])
    for experiment_config in _expanded_configs(config):
        key, observable_key = random.split(key)
        key, experiment_key = random.split(key)
        experiments.append(
            _build_experiment(
                experiment_config,
                observable_key,
                experiment_key,
                repetitions,
            )
        )
    return ExperimentPlan(
        experiments,
        _resolved_section(config, "plot"),
    )


def _plot_results(experiments: Experiments, plot: Mapping[str, Any]) -> None:
    options = dict(plot)
    if not options.pop("enabled", True):
        return
    method = options.pop("function", "plot_avg_var")
    if method not in PLOT_METHODS:
        raise ValueError(f"Unknown plotting function {method!r}")
    path = options.pop("path", None)
    options.pop("name_dict", None)
    options.update(options.pop("style", {}) or {})
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    getattr(experiments, method)(path_save=path, **options)


def run_from_config(
    cfg: DictConfig | Mapping[str, Any] | str | Path,
) -> Experiments:
    """Run, cache, aggregate, and plot a configuration.

    Args:
        cfg: YAML path, OmegaConf configuration, or equivalent mapping.

    Returns:
        The populated experiment collection used to produce the plot.

    Notes:
        An existing `<path_prefix>.npy` is loaded as an exact cache hit without
        validating metadata against the configuration.

    Examples:
        ```python
        experiments = run_from_config("cfgs/xy_local_n12.yaml")
        ```
    """

    plan = parse_experiment_config(cfg)
    for experiment in plan.experiments:
        if experiment.path_save is not None:
            Path(experiment.path_save).parent.mkdir(parents=True, exist_ok=True)
    experiments = Experiments(
        plan.experiments,
        name_dict=plan.plot.get("name_dict", {}),
    )
    _plot_results(experiments, plan.plot)
    return experiments


def _normalise_cli_config_alias() -> None:
    """Accept ``--config`` as a short alias for Hydra's ``--config-name``."""

    for index, argument in enumerate(sys.argv):
        if argument == "--config":
            sys.argv[index] = "--config-name"
        elif argument.startswith("--config="):
            sys.argv[index] = argument.replace("--config=", "--config-name=", 1)


_normalise_cli_config_alias()


@hydra_main(
    version_base="1.3",
    config_path=(
        None if any(arg.startswith("--config-dir") for arg in sys.argv) else "cfgs"
    ),
    config_name="xy_local_n12",
)
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


if __name__ == "__main__":
    main()
