# Architecture

The codebase is deliberately a small collection of root-level modules and a
`tools` namespace. The documentation follows that layout; it does not require
installing or importing the project as a package.

## Execution pipeline

1. `run_experiment.py` composes a Hydra configuration and expands experiment
   sweeps.
2. It constructs state, observable, and shadow objects for each qubit count.
3. `ShadowScalingExperiment` loads an exact cache hit or delegates to the
   streamed JAX batch runner.
4. A shadow backend samples measurement outcomes and reconstructs observable
   estimates.
5. `Experiments` aggregates errors over observables and state repetitions.
6. `plotting.py` produces the configured PDF.

## Main layers

| Layer | Modules | Responsibility |
| --- | --- | --- |
| Configuration | `run_experiment.py` | Parse YAML, expand sweeps, construct experiments |
| Orchestration | `evaluate_shadow.py`, `tools/batching.py` | Execute, batch, cache, and load results |
| Protocols | `shadow.py`, `tools/majorana.py` | Sampling circuits and inverse channels |
| Models | `tools/state.py`, `tools/observable.py` | State and observable representations |
| Simulation | `tools/statevector.py`, `tools/clifford.py` | Direct statevector and tableau kernels |
| Estimation | `tools/estimator.py` | Streaming mean and median-of-means estimators |
| Presentation | `experiment.py`, `plotting.py` | Aggregate metrics and render figures |

## Backend boundary

`Shadow.sample()` selects one of three backends while the protocol classes keep
the same external lifecycle:

- `statevector` calls specialized JAX sampling kernels.
- `clifford` evolves and samples stabilizer tableaus.
- `pennylane` builds Catalyst-compiled PennyLane circuits.

All backends feed the same snapshot reconstruction and estimator interfaces.
This is why configurations can change the simulator without changing the
experiment orchestration.

## Streaming and memory

The batch runner materializes weak properties for one sample batch rather than
the entire maximum sample count. Estimator state retains cumulative sums and
the requested prefix values. This bounds the dominant temporary allocation by
`batch.states × observable_count × batch.samples`.

## Result contract

Each result file contains one `float32` array with axes:

```text
(ground_truth_or_prediction, qubit_count, state_repetition, observable, sample_count)
```

Index `0` is ground truth and index `1` is the prediction. Ground truth is
repeated over the final axis so that both tensors have the same shape.

## Documentation pipeline

API documentation is derived from the same source that users execute:

```text
Python source
    ├── annotations: accepted and returned types
    └── Google-style docstrings: behavior and semantics
                    │
                    ▼
              Griffe AST model
                    │
                    ▼
          mkdocstrings-python rendering
                    │
                    ▼
              Zensical website
```

Griffe inspection is disabled, so building the site never initializes JAX,
Catalyst, PennyLane, or a simulator backend.
