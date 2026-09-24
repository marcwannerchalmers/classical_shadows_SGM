# Run an experiment

This tutorial creates a small configuration without modifying the checked-in
research workloads.

## 1. Copy a configuration

```bash
cp cfgs/xy_local_n12.yaml cfgs/xy_local_quickstart.yaml
```

In the copy, reduce the sample axis and repetitions:

```yaml
samples:
  powers: {base: 2, start: 5, stop: 9}
repetitions: 2
```

Set `observables.count` to `20`, then choose compatible batches:

```yaml
batch:
  states: 1
  observables: 10
  samples: 32
```

## 2. Use fresh output paths

Change every experiment's `path_prefix`, for example:

```yaml
path_prefix: results/quickstart_sgm_uniform
```

Also choose a new plot path:

```yaml
plot:
  path: plots/quickstart.pdf
```

Fresh names ensure that the run neither loads the checked-in caches nor
overwrites them.

## 3. Run

```bash
python run_experiment.py --config-dir=cfgs --config=xy_local_quickstart.yaml
```

The first execution includes JAX and Catalyst compilation. Later executions
load matching result arrays and only regenerate the plot.

## 4. Inspect the result

```python
from pathlib import Path

import numpy as np

path = Path("results/quickstart_sgm_uniform.npy")
ground_truth, predictions = np.load(path)
print(ground_truth.shape, predictions.shape)
```

The last axis follows the configured sample-count sequence.
