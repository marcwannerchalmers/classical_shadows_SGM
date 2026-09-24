"""Seaborn-based plots for shadow-scaling experiments."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


Metric = Literal["mae", "rmse", "max"]
XAxis = Literal["N", "n"]

_METRIC_LABELS = {
    "mae": "Mean absolute error",
    "rmse": "RMSE",
    "max": "Maximum absolute error",
}
_METRIC_ALIASES = {
    "mae": "mae",
    "mean_absolute_error": "mae",
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "max": "max",
    "max_error": "max",
    "maximum_error": "max",
}
_MARKERS = ("o", "X", "s", "P", "D", "v", "^", "<", ">", "*")


def _font_size(font_sizes, name: str, default: float = 10) -> float:
    if font_sizes is None:
        return default
    if isinstance(font_sizes, (int, float)):
        return float(font_sizes)
    return float(font_sizes.get(name, default))


def _normalise_metric(metric: str) -> Metric:
    key = metric.lower().strip().replace(" ", "_")
    try:
        return _METRIC_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown metric {metric!r}; choose one of: mae, max, rmse"
        ) from exc


def _new_axes(aspect_ratio: float | None, *, wide: bool = False):
    sns.set_theme(style="whitegrid")
    height = 5.5
    width = (9.5 if wide else 7.5) if aspect_ratio is None else aspect_ratio * height
    return plt.subplots(figsize=(width, height))


def _save(fig, path_save: str | Path | None) -> None:
    fig.tight_layout()
    if path_save is not None:
        fig.savefig(path_save, bbox_inches="tight", dpi=300)


def _legend(ax, font_sizes, location=None, *, handles=None, split=False) -> None:
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [handle.get_label() for handle in handles]
    kwargs = {"loc": location or "best"}
    if split and location is None:
        kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 1.0),
            "borderaxespad": 0.0,
        }
    ax.legend(
        handles,
        labels,
        title="Shadow model / segment" if split else "Shadow model",
        fontsize=_font_size(font_sizes, "legend"),
        title_fontsize=_font_size(font_sizes, "legend_title"),
        frameon=True,
        **kwargs,
    )


def _series_metadata(plot_df: pd.DataFrame, x_axis: XAxis):
    """Add stable labels and Seaborn palette/marker mappings."""

    fixed_axis: XAxis = "n" if x_axis == "N" else "N"
    keys = [
        key
        for key, _ in plot_df.groupby(
            ["Shadow model", fixed_axis, "experiment_name"],
            sort=True,
            dropna=False,
        )
    ]
    counts = {
        name: sum(key[2] == name for key in keys)
        for name in dict.fromkeys(key[2] for key in keys)
    }
    labels = {
        key: (
            f"{key[2]}, {fixed_axis}={key[1]}"
            if counts[key[2]] > 1
            else str(key[2])
        )
        for key in keys
    }
    result = plot_df.copy()
    result["_series"] = [
        labels[(row["Shadow model"], row[fixed_axis], row["experiment_name"])]
        for _, row in result.iterrows()
    ]
    order = [labels[key] for key in keys]
    experiment_names = list(dict.fromkeys(key[2] for key in keys))
    colors = dict(
        zip(experiment_names, sns.color_palette("tab10", len(experiment_names)))
    )
    palette = {labels[key]: colors[key[2]] for key in keys}
    markers = {
        label: _MARKERS[index % len(_MARKERS)]
        for index, label in enumerate(order)
    }
    return result, order, palette, markers


def _long_repetitions(plot_df: pd.DataFrame) -> pd.DataFrame:
    result = plot_df.copy()
    result["repetition_metric"] = result["repetition_metric"].map(
        lambda values: np.atleast_1d(values).tolist()
    )
    result = result.explode("repetition_metric", ignore_index=True)
    result["repetition_metric"] = result["repetition_metric"].astype(float)
    return result


def _contiguous_runs(group: pd.DataFrame, x_axis: XAxis) -> Iterator[pd.DataFrame]:
    group = group.sort_values(x_axis).reset_index(drop=True)
    run_ids = group["all_predictions_zero"].ne(
        group["all_predictions_zero"].shift()
    ).cumsum()
    yield from (run for _, run in group.groupby(run_ids, sort=False))


def _configure_metric_axes(
    ax,
    plot_df: pd.DataFrame,
    metric: Metric,
    x_axis: XAxis,
    xlog: bool,
    ylog: bool,
    font_sizes,
    x_tick_stride: int,
) -> None:
    if x_tick_stride < 1:
        raise ValueError("x_tick_stride must be at least 1")
    if xlog:
        ax.set_xscale("log", base=2)
    if xlog or x_tick_stride > 1:
        ticks = np.sort(plot_df[x_axis].unique())[::x_tick_stride]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [rf"$2^{{{int(np.log2(value))}}}$" for value in ticks]
            if x_axis == "N" and xlog
            else [str(value) for value in ticks]
        )

    positive_means = plot_df.loc[plot_df["mean"] > 0, "mean"]
    if ylog:
        if positive_means.empty:
            raise ValueError("A logarithmic y-axis requires a positive value")
        ax.set_yscale("log")
        positive_lower = plot_df.loc[plot_df["ci_low"] > 0, "ci_low"]
        ymin = min(
            positive_means.min(),
            positive_lower.min() if not positive_lower.empty else np.inf,
        ) * 0.8
    else:
        ymin = 0.0
    finite_upper = plot_df.loc[np.isfinite(plot_df["ci_high"]), "ci_high"]
    ymax = finite_upper.max() if not finite_upper.empty else plot_df["mean"].max()
    ax.set_ylim(ymin, ymax * (1.2 if ylog else 1.05))
    ax.set_xlabel(
        "Number of samples $N$" if x_axis == "N" else "Number of qubits $n$",
        fontsize=_font_size(font_sizes, "axes"),
    )
    ax.set_ylabel(_METRIC_LABELS[metric], fontsize=_font_size(font_sizes, "axes"))
    ax.tick_params(labelsize=_font_size(font_sizes, "ticks"))


def _plot_metric(
    plot_df: pd.DataFrame,
    path_save: str | Path | None = None,
    xlog: bool = False,
    ylog: bool = False,
    metric: str = "mae",
    x: str = "N",
    font_sizes=None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
    aspect_ratio: float | None = None,
    x_tick_stride: int = 1,
    *,
    split_zero_predictions: bool = False,
):
    metric_name = _normalise_metric(metric)
    if x not in {"N", "n"}:
        raise ValueError(f"Unknown x-axis {x!r}; choose 'N' or 'n'")
    x_axis: XAxis = x
    if x_axis == "n":
        groups = ["Shadow model", "n", "experiment_name"]
        plot_df = plot_df[
            plot_df["N"] == plot_df.groupby(groups, dropna=False)["N"].transform("max")
        ]
    plot_df, order, palette, markers = _series_metadata(plot_df, x_axis)
    fig, ax = _new_axes(aspect_ratio, wide=split_zero_predictions)
    line_args = {
        "x": x_axis,
        "y": "repetition_metric",
        "estimator": "mean",
        "sort": True,
        "linewidth": line_width,
        "markersize": marker_size,
        "ax": ax,
    }

    if not split_zero_predictions:
        sns.lineplot(
            data=_long_repetitions(plot_df),
            hue="_series",
            style="_series",
            hue_order=order,
            style_order=order,
            palette=palette,
            markers=markers,
            dashes=False,
            errorbar="sd",
            **line_args,
        )
        _legend(ax, font_sizes, legend_location)
    else:
        handles = []
        for series in order:
            group = plot_df[plot_df["_series"] == series]
            handles.append(
                Line2D(
                    [], [], label=series, color=palette[series],
                    marker=markers[series], linewidth=line_width,
                    markersize=marker_size,
                )
            )
            for run in _contiguous_runs(group, x_axis):
                all_zero = bool(run["all_predictions_zero"].iloc[0])
                sns.lineplot(
                    data=_long_repetitions(run),
                    color=palette[series],
                    marker=markers[series],
                    linestyle="--" if all_zero else "-",
                    errorbar=None if all_zero else "sd",
                    legend=False,
                    **line_args,
                )
        handles += [
            Line2D([], [], color="0.35", label="At least one nonzero prediction"),
            Line2D([], [], color="0.35", linestyle="--", label="All predictions zero"),
        ]
        _legend(ax, font_sizes, legend_location, handles=handles, split=True)

    _configure_metric_axes(
        ax, plot_df, metric_name, x_axis, xlog, ylog, font_sizes, x_tick_stride
    )
    _save(fig, path_save)
    return fig, ax


def plot_avg_var(
    plot_df: pd.DataFrame, **kwargs: Any
) -> tuple[Figure, Axes]:
    """Plot an aggregated error metric against samples or qubits.

    Args:
        plot_df: Aggregated rows produced by `Experiments.get_plot_dataframe`.
        **kwargs: Axis, metric, output, and style options accepted by the shared
            metric renderer.

    Returns:
        The Matplotlib figure and axes.
    """

    return _plot_metric(plot_df, **kwargs)


def plot_avg_var_zero_segments(
    plot_df: pd.DataFrame, **kwargs: Any
) -> tuple[Figure, Axes]:
    """Plot errors with all-zero runs dashed and without shading.

    Args:
        plot_df: Aggregated rows produced by `Experiments.get_plot_dataframe`.
        **kwargs: Axis, metric, output, and style options accepted by the shared
            metric renderer.

    Returns:
        The Matplotlib figure and axes.
    """

    return _plot_metric(plot_df, split_zero_predictions=True, **kwargs)


def _sample_complexity_data(
    plot_df: pd.DataFrame, epsilon: float, remove_zero_predictions: bool
) -> pd.DataFrame:
    rows = []
    group_cols = ["Shadow model", "n", "experiment_name"]
    for keys, group in plot_df.groupby(group_cols, sort=False, dropna=False):
        group = group.sort_values("N")
        sample_counts = group["N"].to_numpy()
        below = np.stack(group["repetition_metric"]) < epsilon
        if remove_zero_predictions:
            below &= ~np.stack(group["zero_by_repetition"])
        reached = below.any(axis=0)
        first = np.argmax(below, axis=0)
        thresholds = np.full(len(reached), sample_counts[-1], dtype=float)
        thresholds[reached] = sample_counts[first[reached]]
        rows.extend((*keys, value, int(reached.sum())) for value in thresholds)

    result = pd.DataFrame(rows, columns=group_cols + ["N_threshold", "reached"])
    kept = []
    for _, group in result.groupby(
        ["Shadow model", "experiment_name"], sort=True, dropna=False
    ):
        crossing_by_n = group.groupby("n", sort=True)["reached"].first()
        no_crossing = crossing_by_n.index[crossing_by_n == 0]
        if len(no_crossing):
            group = group[group["n"] < no_crossing[0]]
        kept.append(group)
    return pd.concat(kept, ignore_index=True)


def plot_sample_complexity(
    plot_df: pd.DataFrame,
    epsilon: float,
    path_save: str | Path | None = None,
    xlog: bool = False,
    ylog: bool = True,
    metric: str = "rmse",
    x: str = "n",
    font_sizes: float | dict[str, float] | None = None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
    remove_zero_predictions: bool = False,
    aspect_ratio: float | None = None,
    x_tick_stride: int = 1,
) -> tuple[Figure, Axes]:
    """Plot the first strict error-threshold crossing over qubit count.

    Args:
        plot_df: Aggregated rows produced by `Experiments.get_plot_dataframe`.
        epsilon: Strict upper error threshold.
        path_save: Optional figure output path.
        xlog: Whether to use a logarithmic x-axis.
        ylog: Whether to use a logarithmic y-axis.
        metric: Error reduction: `mae`, `rmse`, or `max`.
        x: Horizontal axis. This plot requires `n`.
        font_sizes: Uniform font size or sizes keyed by plot element.
        marker_size: Marker size in points.
        line_width: Line width in points.
        legend_location: Matplotlib legend location, or `None` for automatic
            placement.
        remove_zero_predictions: Ignore crossings where every prediction is
            exactly zero.
        aspect_ratio: Figure width divided by height.
        x_tick_stride: Keep every nth candidate x tick.

    Returns:
        The Matplotlib figure and axes.

    Raises:
        ValueError: If `x` is not `n` or no state repetition reaches the
            threshold.
    """

    if x != "n":
        raise ValueError("Sample-complexity plots require x: n")
    metric_name = _normalise_metric(metric)
    threshold_df = _sample_complexity_data(
        plot_df, float(epsilon), remove_zero_predictions
    )
    if threshold_df.empty:
        raise ValueError(f"No state repetition reaches epsilon={epsilon}")

    labels = [
        str(key[1])
        for key in threshold_df.groupby(
            ["Shadow model", "experiment_name"], sort=True, dropna=False
        ).groups
    ]
    threshold_df["_series"] = threshold_df["experiment_name"].astype(str)
    palette = dict(zip(labels, sns.color_palette("tab10", len(labels))))
    markers = {
        label: _MARKERS[index % len(_MARKERS)]
        for index, label in enumerate(labels)
    }

    fig, ax = _new_axes(aspect_ratio)
    sns.lineplot(
        data=threshold_df,
        x="n",
        y="N_threshold",
        hue="_series",
        style="_series",
        hue_order=labels,
        style_order=labels,
        palette=palette,
        markers=markers,
        dashes=False,
        estimator="mean",
        errorbar="sd",
        linewidth=line_width,
        markersize=marker_size,
        sort=True,
        ax=ax,
    )
    if x_tick_stride < 1:
        raise ValueError("x_tick_stride must be at least 1")
    if xlog:
        ax.set_xscale("log", base=2)
    if x_tick_stride > 1:
        ax.set_xticks(np.sort(threshold_df["n"].unique())[::x_tick_stride])
    if ylog:
        ax.set_yscale("log", base=2)
    ymax = threshold_df.groupby(["_series", "n"])["N_threshold"].mean().max()
    ax.set_ylim(top=ymax * 1.05)
    ax.set_xlabel("Number of qubits $n$", fontsize=_font_size(font_sizes, "axes"))
    ax.set_ylabel(
        rf"Mean smallest $N$ with {_METRIC_LABELS[metric_name]} $< {epsilon:g}$",
        fontsize=_font_size(font_sizes, "axes"),
    )
    ax.tick_params(labelsize=_font_size(font_sizes, "ticks"))
    _legend(ax, font_sizes, legend_location)
    _save(fig, path_save)
    return fig, ax
