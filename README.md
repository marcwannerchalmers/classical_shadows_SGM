# Classical shadows with SGM

This repository simulates and compares classical-shadow protocols, with a
focus on selective global measurements (SGM). Experiments are described by
Hydra YAML files and can compare:

- SGM shadows with uniform, binomial, or optimized fixed-weight settings;
- local Pauli shadows;
- global Clifford shadows; and
- fermionic Gaussian shadows (FGU) for Majorana observables.

The implementation includes direct JAX statevector kernels, an integer
stabilizer-tableau backend, streamed batching for large sample counts, result
caching, and Seaborn/Matplotlib plots.

## Installation

Python 3.13 is the tested version. Create an isolated environment from the
repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The Catalyst dependency officially supports macOS and Linux; Linux is required
for the NVIDIA GPU setup described below.

Verify the installation:

```bash
python -c "import jax, pennylane, catalyst; print(jax.devices())"
python run_experiment.py --help
```

The versions in `requirements.txt` reproduce the environment used to create
the checked-in results. If a pinned wheel is unavailable on a different Python
or operating-system version, use Python 3.13 before changing dependency
versions.

## Running experiments

Run a configuration from the repository root:

```bash
python run_experiment.py --config-dir=cfgs --config=xy_local_n12.yaml
```

`--config` is a convenience alias for Hydra's native `--config-name` option,
so the following is equivalent:

```bash
python run_experiment.py --config-dir=cfgs --config-name=xy_local_n12.yaml
```

Each configuration runs all entries under `experiments`, writes arrays below
`results/`, and creates its configured PDF below `plots/`. Parent directories
are created automatically.

The repository contains these configurations:

| Configuration | Workload | Backends |
| --- | --- | --- |
| `xy_local_n12.yaml` | 12-qubit Haar-random states and 500 local XY observables | JAX statevector |
| `xy_local_n20.yaml` | 20-qubit Haar-random states and 50 local XY observables | JAX statevector |
| `majorana_n12.yaml` | All order-2 Majorana RDM observables at 12 qubits | JAX statevector |
| `majorana_sample_comp.yaml` | Majorana sample-complexity sweep from 4 to 20 qubits | JAX statevector |
| `random_GHZ.yaml` | GHZ-type stabilizer sample-complexity sweep | JAX tableau |

These are research-scale configurations: some request more than one million
samples and can run for a long time. Checked-in `.npy` arrays are treated as a
cache, so an unmodified command normally loads them and regenerates the plot.
To recompute one experiment, rename or remove only that experiment's exact
`path_prefix.npy`, or change its `path_prefix` in a copied YAML file. Existing
cache files are loaded without checking that their contents match the current
configuration.

For a small trial run, copy a configuration and reduce `samples`,
`repetitions`, and the observable `count`. Give every experiment in the copy a
new `path_prefix` so it cannot load or overwrite a full cached result.

## Configuration reference

The top-level experiment axes accept three forms. Every `stop` is exclusive:

```yaml
qubits:
  values: [8, 10, 12]

samples:
  range: {start: 1024, stop: 8193, step: 1024}

# or
samples:
  powers: {base: 2, start: 7, stop: 21}  # 2^7 through 2^20
```

The principal sections are:

- `seed` and `repetitions`: random seed and number of independently sampled
  states.
- `state.class`: `HRState`, `GHZType`, or `MajoranaState`.
- `observables`: random Pauli words, a Majorana RDM family, or an external
  `.npy` array. Pauli labels are `I`, `X`, `Y`, and `Z`.
- `batch`: state, observable, and sample batch sizes.
- `experiments`: shadow class, simulator, SGM distribution, display name,
  result prefix, and logging choice for each method.
- `plot`: plotting function, error metric, output path, axes, and style.

Supported shadow classes are `SGMShadow`, `PauliShadow`, `CliffordShadow`, and
`FermionicGaussianShadow`. The simulator choices are:

- `statevector`: specialized JAX kernels for explicit statevectors;
- `clifford`: the JAX stabilizer-tableau simulator, intended for stabilizer
  states such as the GHZ workload; and
- `pennylane`: the general PennyLane/Catalyst circuit fallback.

An SGM distribution is configured under `shadow_args`:

```yaml
shadow_args:
  simulator: statevector
  distribution:
    name: binomial
    q: 0.9
```

`name` may be `uniform`, `binomial`, or `fixed_weight`. A binomial `q` may be a
number, `mean_xy_support`, or `rdm_optim`; optimized fixed-weight settings use
`weights: rdm_optim`. See the existing Majorana configurations for complete
examples. Optional depolarizing noise is written as
`noise: {name: depolarizing, p: 0.1}`.

When batching is enabled, the following divisibility constraints must hold:

```text
max(samples) % batch.samples == 0
repetitions % batch.states == 0
observable_count % batch.observables == 0
```

Lower batch sizes reduce peak memory use but can increase loop overhead and
compilation time.

## Results and plots

For a configured prefix such as `results/my_experiment`, NumPy writes
`results/my_experiment.npy`. The stored `float32` array has axes:

```text
(ground_truth_or_prediction, qubit_count, state_repetition, observable, sample_count)
```

Index `0` contains ground truth and index `1` contains predictions. Ground
truth is repeated across the final axis. Plotting aggregates the selected
metric over observables for each repetition, then plots the mean and sample
standard deviation across repetitions.

Available plot functions are `plot_avg_var`, `plot_avg_var_zero_segments`, and
`plot_sample_complexity`. Supported metrics are `mae`, `rmse`, and `max`.

## GPU execution

### JAX kernels (recommended)

The `statevector` and `clifford` backends are written in JAX. They automatically
execute on the default JAX accelerator, so the shipped YAML files need no code
or simulator change after a GPU-enabled JAX installation. The statevector
backend is the main GPU target; benchmark the integer-heavy tableau backend,
because a GPU is not necessarily faster for every stabilizer workload.

The supported NVIDIA path is Linux on x86-64 or aarch64. For the repository's
pinned JAX 0.7.1 environment, CUDA 12 is the conservative choice. It requires
an NVIDIA driver at least version 525; JAX 0.7.1's CUDA build supports CUDA
12.1 or newer and requires cuDNN 9.10.2 or newer when using a local CUDA
installation. JAX recommends its pip-provided CUDA libraries, which avoid a
separate CUDA toolkit installation.

In a fresh Python 3.13 environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade "jax[cuda12]==0.7.1"
```

Confirm that JAX sees a GPU before starting a long experiment:

```bash
nvidia-smi
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
```

Then run a configuration normally:

```bash
python run_experiment.py --config-dir=cfgs --config=xy_local_n12.yaml
```

Remember that an existing result file skips computation. Change the copied
configuration's `path_prefix` when checking that a workload really executes on
the GPU.

JAX is CPU-only on macOS in its officially supported installation matrix;
Apple's experimental plugin is not part of this repository's tested Catalyst
stack. Use a Linux NVIDIA machine for the documented GPU route.

### Optional PennyLane Lightning-GPU device

The direct JAX samplers still use PennyLane/Catalyst for non-tableau ground
truth evaluation, and `simulator: pennylane` uses it for sampling as well. To
place those circuit operations on an NVIDIA GPU, install the version-matched
Lightning-GPU plugin:

```bash
python -m pip install "pennylane-lightning-gpu==0.45.0"
python -c "import pennylane as qml; print(qml.device('lightning.gpu', wires=2))"
```

Add the device to `shadow_args` for each experiment that should use it:

```yaml
shadow_args:
  simulator: statevector
  device: lightning.gpu
```

With `simulator: statevector`, this changes the PennyLane ground-truth/fallback
device while the specialized sampler remains a JAX GPU kernel. To run the
general compiled circuit sampler on Lightning-GPU instead, set both:

```yaml
shadow_args:
  simulator: pennylane
  device: lightning.gpu
```

Lightning-GPU uses NVIDIA cuQuantum and is a Linux device. Catalyst supports
`lightning.gpu`, but plugin, CUDA, Catalyst, and PennyLane versions must remain
compatible.

### GPU memory and troubleshooting

- If an experiment runs out of memory, lower `batch.samples` first, then
  `batch.observables` or `batch.states`, while preserving the divisibility
  rules above.
- JAX preallocates GPU memory for performance. On a shared GPU, try
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` or set a smaller
  `XLA_PYTHON_CLIENT_MEM_FRACTION` before launching the process.
- If `jax.devices()` reports only CPU, check the NVIDIA driver and reinstall
  the CUDA extra after `requirements.txt`. With pip-managed CUDA libraries,
  an incompatible `LD_LIBRARY_PATH` can make JAX load the wrong libraries.
- The first run for a new combination of shapes and static settings includes
  JIT compilation and can be much slower than subsequent execution.

Official references: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html),
[PennyLane Lightning-GPU](https://pennylane.ai/devices/lightning-gpu), and
[PennyLane compiled workflows](https://docs.pennylane.ai/en/stable/introduction/compiling_workflows.html).

## Code layout

```text
run_experiment.py       Hydra entry point and YAML parser
evaluate_shadow.py      execution, batching, caching, and .npy persistence
experiment.py           experiment collection and result aggregation
plotting.py             plotting functions
shadow.py               SGM, Pauli, and Clifford shadow protocols
tools/batching.py       streamed JAX batch runner
tools/majorana.py       Majorana states, observables, and FGU shadows
tools/statevector.py    direct statevector sampling kernels
tools/clifford.py       stabilizer-tableau operations
tools/state.py          Haar-random and GHZ-type states
tools/observable.py     observable representations
tools/distributions.py  SGM setting distributions and optimizers
tools/estimator.py      streaming mean and median-of-means estimators
cfgs/                   runnable experiment configurations
results/                cached numerical outputs
plots/                  generated figures
```

For programmatic use, `run_from_config()` accepts a YAML path, an OmegaConf
configuration, or a mapping:

```python
from run_experiment import run_from_config

experiments = run_from_config("cfgs/xy_local_n12.yaml")
```

`parse_experiment_config()` performs configuration parsing without running the
experiments.

## Documentation

Install the static documentation toolchain separately from the scientific
runtime dependencies:

```bash
python -m pip install -r requirements-docs.txt
zensical serve
```

Build and validate the complete site with:

```bash
zensical build --strict
```

The site source is under `docs/`. API pages are rendered by
mkdocstrings-python from Griffe's static source model, with Google-style
docstrings and Python annotations as the source of signature types. The build
does not import the runtime modules.

## License

This project is released under the MIT License. See `LICENSE`.
