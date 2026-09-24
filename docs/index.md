# Classical shadows with SGM

This project simulates and compares classical-shadow measurement protocols,
with an emphasis on selective global measurements (SGM). Experiment plans are
declared in YAML and executed with JAX-backed statevector or stabilizer-tableau
simulators.

Use the documentation in three layers:

- [Getting started](getting-started.md) covers installation, a first run, and
  GPU setup.
- [Concepts](concepts/architecture.md) explain the execution pipeline and the
  configuration model.
- [API reference](reference/api.md) is generated from type annotations and
  Google-style docstrings by Griffe and mkdocstrings.

## Supported protocols

- Selective global measurement shadows with uniform, binomial, and optimized
  fixed-weight distributions.
- Product-Pauli shadows.
- Global-Clifford shadows.
- Fermionic Gaussian shadows for Majorana observables.

## Typical workflow

```text
YAML configuration
        │
        ▼
parse_experiment_config()
        │
        ▼
ShadowScalingExperiment ── load cached .npy or run JAX simulation
        │
        ▼
Experiments ── aggregate metrics ── write PDF plot
```

Continue with [Getting started](getting-started.md).
