"""Experiment collections, metric aggregation, and plotting dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from plotting import (
    _normalise_metric,
    plot_avg_var as plot_avg_var_figure,
    plot_avg_var_zero_segments as plot_avg_var_zero_segments_figure,
    plot_sample_complexity as plot_sample_complexity_figure,
)
from evaluate_shadow import ShadowScalingExperiment

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Experiments:
    """Run, collect, and plot a group of shadow-scaling experiments.

    Args:
        scaling_experiments: Experiments to run immediately, in display order.
        name_dict: Optional mapping from experiment names to legend labels.

    Notes:
        Construction executes each experiment. Cached result files may make
        this a load-only operation.
    """

    def __init__(
        self,
        scaling_experiments: list[ShadowScalingExperiment],
        name_dict: Mapping[str, str] | None = None,
    ) -> None:
        self.experiments = scaling_experiments
        self.name_dict = name_dict or {}
        for experiment in scaling_experiments:
            experiment.run()

    def plot_avg_var(
        self,
        path_save: str | Path | None = None,
        xlog: bool = False,
        ylog: bool = False,
        metric: str = "mae",
        x: str = "N",
        aggregation_batch_size: int = 16,
        progress: bool = True,
        N_min: int | None = None,
        N_max: int | None = None,
        n_min: int | None = None,
        n_max: int | None = None,
        **style: Any,
    ) -> tuple[Figure, Axes]:
        """Plot an error metric against sample or qubit count.

        Args:
            path_save: Optional figure output path.
            xlog: Whether to use a base-2 logarithmic x-axis.
            ylog: Whether to use a logarithmic y-axis.
            metric: Error reduction: `mae`, `rmse`, or `max`.
            x: Horizontal axis, either `N` or `n`.
            aggregation_batch_size: Number of sample-count positions processed
                together while constructing plot data.
            progress: Whether to show aggregation progress.
            N_min: Optional inclusive lower sample-count bound.
            N_max: Optional inclusive upper sample-count bound.
            n_min: Optional inclusive lower qubit-count bound.
            n_max: Optional inclusive upper qubit-count bound.
            **style: Plot styling accepted by `plotting.plot_avg_var`.

        Returns:
            The Matplotlib figure and axes.

        Raises:
            ValueError: If the metric, axis, batch size, or bounds are invalid.
        """

        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_avg_var_figure(
            data,
            path_save=path_save,
            x=x,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            **style,
        )

    def plot_avg_var_zero_segments(
        self,
        path_save: str | Path | None = None,
        xlog: bool = False,
        ylog: bool = False,
        metric: str = "mae",
        x: str = "N",
        aggregation_batch_size: int = 16,
        progress: bool = True,
        N_min: int | None = None,
        N_max: int | None = None,
        n_min: int | None = None,
        n_max: int | None = None,
        **style: Any,
    ) -> tuple[Figure, Axes]:
        """Plot errors while distinguishing all-zero prediction segments.

        Args:
            path_save: Optional figure output path.
            xlog: Whether to use a base-2 logarithmic x-axis.
            ylog: Whether to use a logarithmic y-axis.
            metric: Error reduction: `mae`, `rmse`, or `max`.
            x: Horizontal axis, either `N` or `n`.
            aggregation_batch_size: Number of sample-count positions processed
                together while constructing plot data.
            progress: Whether to show aggregation progress.
            N_min: Optional inclusive lower sample-count bound.
            N_max: Optional inclusive upper sample-count bound.
            n_min: Optional inclusive lower qubit-count bound.
            n_max: Optional inclusive upper qubit-count bound.
            **style: Plot styling accepted by the zero-segment renderer.

        Returns:
            The Matplotlib figure and axes.
        """

        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_avg_var_zero_segments_figure(
            data,
            path_save=path_save,
            x=x,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            **style,
        )

    def plot_sample_complexity(
        self,
        epsilon: float,
        path_save: str | Path | None = None,
        xlog: bool = False,
        ylog: bool = True,
        metric: str = "rmse",
        x: str = "n",
        aggregation_batch_size: int = 16,
        progress: bool = True,
        N_min: int | None = None,
        N_max: int | None = None,
        n_min: int | None = None,
        n_max: int | None = None,
        remove_zero_predictions: bool = False,
        **style: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the first strict error-threshold crossing for each qubit count.

        Args:
            epsilon: Strict upper error threshold.
            path_save: Optional figure output path.
            xlog: Whether to use a logarithmic x-axis.
            ylog: Whether to use a logarithmic y-axis.
            metric: Error reduction: `mae`, `rmse`, or `max`.
            x: Horizontal axis. Sample-complexity plots require `n`.
            aggregation_batch_size: Number of sample-count positions processed
                together while constructing plot data.
            progress: Whether to show aggregation progress.
            N_min: Optional inclusive lower sample-count bound.
            N_max: Optional inclusive upper sample-count bound.
            n_min: Optional inclusive lower qubit-count bound.
            n_max: Optional inclusive upper qubit-count bound.
            remove_zero_predictions: Ignore threshold crossings caused by an
                all-zero prediction vector.
            **style: Plot styling accepted by the sample-complexity renderer.

        Returns:
            The Matplotlib figure and axes.

        Raises:
            ValueError: If no repetition reaches the threshold or the x-axis
                is not `n`.
        """

        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_sample_complexity_figure(
            data,
            epsilon=float(epsilon),
            path_save=path_save,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            x=x,
            remove_zero_predictions=remove_zero_predictions,
            **style,
        )

    def get_plot_dataframe(
        self,
        metric: str = "mae",
        N_batch_size: int = 16,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Aggregate plot statistics directly from result tensors.

        Args:
            metric: Error reduction: `mae`, `rmse`, or `max`.
            N_batch_size: Number of sample-count positions transferred from
                JAX together.
            progress: Whether to show aggregation progress.

        Returns:
            One row per experiment, qubit count, and sample count, including
            repetition-level metrics and summary statistics.

        Raises:
            ValueError: If `N_batch_size` is not positive.
        """

        metric_name = _normalise_metric(metric)
        if N_batch_size <= 0:
            raise ValueError("aggregation_batch_size must be positive")
        total = sum(
            len(experiment.ns)
            * ((len(experiment.Ns) + N_batch_size - 1) // N_batch_size)
            for experiment in self.experiments
        )
        rows = []
        with tqdm(
            total=total,
            desc="Aggregating plot data",
            unit="batch",
            disable=not progress,
        ) as bar:
            for experiment in self.experiments:
                model_name = experiment.shadow_cls.__name__
                Ns = np.asarray(experiment.Ns)
                for n_index, n in enumerate(experiment.ns):
                    for start in range(0, len(Ns), N_batch_size):
                        stop = min(start + N_batch_size, len(Ns))
                        predictions = np.asarray(
                            jax.device_get(
                                experiment.preds[n_index, :, :, start:stop]
                            ),
                            dtype=np.float64,
                        )
                        ground_truth = np.asarray(
                            jax.device_get(
                                experiment.gt[n_index, :, :, start:stop]
                            ),
                            dtype=np.float64,
                        )
                        errors = np.abs(ground_truth - predictions)
                        if metric_name == "mae":
                            repetition_metric = errors.mean(axis=1)
                        elif metric_name == "rmse":
                            repetition_metric = np.sqrt(
                                np.mean(errors**2, axis=1)
                            )
                        else:
                            repetition_metric = errors.max(axis=1)

                        means = repetition_metric.mean(axis=0)
                        stds = (
                            repetition_metric.std(axis=0, ddof=1)
                            if repetition_metric.shape[0] > 1
                            else np.full(stop - start, np.nan)
                        )
                        zero_by_repetition = np.equal(predictions, 0).all(axis=1)
                        all_zero = zero_by_repetition.all(axis=0)
                        for index, N in enumerate(Ns[start:stop]):
                            rows.append(
                                {
                                    "Shadow model": model_name,
                                    "N": int(N),
                                    "n": int(n),
                                    "experiment_name": self.name_dict.get(
                                        experiment.experiment_name,
                                        experiment.experiment_name,
                                    ),
                                    "mean": means[index],
                                    "std": stds[index],
                                    "count": repetition_metric.shape[0],
                                    "repetition_metric": repetition_metric[:, index],
                                    "zero_by_repetition": zero_by_repetition[:, index],
                                    "all_predictions_zero": bool(all_zero[index]),
                                    "ci": stds[index],
                                    "ci_low": means[index] - stds[index],
                                    "ci_high": means[index] + stds[index],
                                }
                            )
                        bar.update()
        result = pd.DataFrame(rows)
        result.attrs["metric"] = metric_name
        return result

    @staticmethod
    def _bounded_plot_data(data, N_min=None, N_max=None, n_min=None, n_max=None):
        for column, lower, upper in (
            ("N", N_min, N_max),
            ("n", n_min, n_max),
        ):
            if lower is not None:
                data = data[data[column] >= lower]
            if upper is not None:
                data = data[data[column] <= upper]
        if data.empty:
            raise ValueError("No plot data remains within the configured bounds")
        return data
