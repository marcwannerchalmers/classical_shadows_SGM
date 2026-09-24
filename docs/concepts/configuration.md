# Configuration

Experiment YAML files live in `cfgs/`. Hydra resolves interpolation before the
configuration is converted into experiment objects.

## Axes

Qubit and sample axes accept explicit values, Python-style ranges, or powers.
Every `stop` is exclusive.

```yaml
qubits:
  values: [8, 10, 12]

samples:
  powers: {base: 2, start: 7, stop: 21}
```

## States and observables

Available state classes are `HRState`, `GHZType`, and `MajoranaState`.
Observables can be random Pauli words, Majorana RDM observables, or arrays
loaded from an `.npy` file. Pauli codes are `I=0`, `X=1`, `Y=2`, and `Z=3`.

## Experiments

Each item under `experiments` selects a shadow class, simulator, cache prefix,
and plot label. For example:

```yaml
- id: sgm_binomial
  shadow_class: SGMShadow
  shadow_args:
    simulator: statevector
    distribution:
      name: binomial
      q: 0.9
  experiment_name: SGM binomial
  path_prefix: results/SGM_example
  verbose: true
```

Available shadow classes are `SGMShadow`, `PauliShadow`, `CliffordShadow`, and
`FermionicGaussianShadow`.

## Batching constraints

The batch runner validates:

```text
max(samples) % batch.samples == 0
repetitions % batch.states == 0
observable_count % batch.observables == 0
```

Lower `batch.samples` first when reducing GPU or CPU memory use. Decrease
`batch.observables` or `batch.states` as needed while maintaining the
divisibility constraints.

## Plot configuration

Supported functions are `plot_avg_var`, `plot_avg_var_zero_segments`, and
`plot_sample_complexity`. The supported metrics are `mae`, `rmse`, and `max`.
Styling belongs under `plot.style`; computation and filtering options remain
direct children of `plot`.

See the checked-in YAML files for complete, reproducible configurations.
