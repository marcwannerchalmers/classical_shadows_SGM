# Getting started

## Requirements

Python 3.13 is the tested version. Run commands from the repository root so
that relative configuration, result, and plot paths resolve consistently.

## CPU installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import jax, pennylane, catalyst; print(jax.devices())"
python run_experiment.py --help
```

## First run

```bash
python run_experiment.py --config-dir=cfgs --config=xy_local_n12.yaml
```

The command writes numerical arrays to `results/` and a PDF to `plots/`.
Existing arrays are caches: if the exact configured `path_prefix.npy` exists,
the simulation is skipped and that array is loaded.

!!! warning

    The shipped configurations are research-scale workloads. Copy a YAML file,
    reduce its sample counts and repetitions, and use new result prefixes for
    an inexpensive first simulation.

## NVIDIA GPU installation

The supported GPU path is Linux with an NVIDIA GPU. The direct `statevector`
and `clifford` simulators are JAX code and automatically use JAX's selected
accelerator.

Install the runtime dependencies and the CUDA 12 extra matching the pinned JAX
version:

```bash
python -m pip install -r requirements.txt
python -m pip install --upgrade "jax[cuda12]==0.7.1"
```

Check device discovery before a long run:

```bash
nvidia-smi
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
```

No YAML change is required for experiments using `simulator: statevector` or
`simulator: clifford`.

### Optional PennyLane GPU device

Install Lightning-GPU when PennyLane/Catalyst circuit execution, including
non-tableau ground-truth evaluation, should also use the GPU:

```bash
python -m pip install "pennylane-lightning-gpu==0.45.0"
```

Then set the device for the relevant experiment:

```yaml
shadow_args:
  simulator: statevector
  device: lightning.gpu
```

Use `simulator: pennylane` instead when the general PennyLane sampler should
replace the specialized JAX sampler.

See the [configuration guide](concepts/configuration.md) for batching and
memory controls.
