from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from easm_cwes_utils import (
    ALL_NODES,
    IMI_NORTH_BOX,
    IMI_WEST_BOX,
    WNPMI_BOX_A,
    WNPMI_BOX_B,
    area_mean,
    find_repo_root,
    make_mmdd_anomaly,
    open_field,
)


WINDOW = 10


def centered_running_mean(df: pd.DataFrame, columns: list[str], window: int = WINDOW) -> pd.DataFrame:
    out = pd.DataFrame({"year": df["year"].to_numpy()})
    for col in columns:
        out[col] = df[col].rolling(window=window, center=True, min_periods=window).mean()
    return out


def significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.05:
        return "*"
    return ""


def linear_trend_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    fit = stats.linregress(x, y)
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "p": float(fit.pvalue),
    }


def quadratic_cycle_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    design = np.column_stack([np.ones_like(x_centered), x_centered, x_centered**2])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ beta
    resid = y - yhat
    n_obs, n_params = design.shape
    dof = n_obs - n_params
    if dof <= 0:
        return {
            "quad_p": np.nan,
            "vertex_year": np.nan,
            "b0": float(beta[0]),
            "b1": float(beta[1]),
            "b2": float(beta[2]),
        }

    s2 = float((resid @ resid) / dof)
    cov = s2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    quad_t = beta[2] / se[2] if se[2] > 0 else np.nan
    quad_p = float(2 * stats.t.sf(np.abs(quad_t), dof)) if np.isfinite(quad_t) else np.nan
    vertex_year = float(-beta[1] / (2 * beta[2]) + x.mean()) if np.abs(beta[2]) > 1e-12 else np.nan
    return {
        "quad_p": quad_p,
        "vertex_year": vertex_year,
        "b0": float(beta[0]),
        "b1": float(beta[1]),
        "b2": float(beta[2]),
    }


def daily_node_index_stats(repo_root: Path, outdir: Path) -> pd.DataFrame:
    assign = pd.read_csv(outdir / "som_daily_assignment.csv")
    assign["date"] = pd.to_datetime(assign["date"]).dt.normalize()

    uwnd = open_field(repo_root / "data" / "processed" / "uwnd_z850_jja_1991_2023.nc", "uwnd")
    uwnd = uwnd.sel(time=uwnd["time"].dt.month.isin([6, 7, 8]))

    if outdir.name == "som_u850":
        clim = uwnd.groupby("time.dayofyear").mean("time")
        uwnd_anom = uwnd.groupby("time.dayofyear") - clim
    else:
        uwnd_anom, _ = make_mmdd_anomaly(uwnd)

    wnpmi = area_mean(uwnd_anom, WNPMI_BOX_A) - area_mean(uwnd_anom, WNPMI_BOX_B)
    imi = area_mean(uwnd_anom, IMI_WEST_BOX) - area_mean(uwnd_anom, IMI_NORTH_BOX)

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(uwnd_anom["time"].values).normalize(),
            "wnpmi": wnpmi.values,
            "imi": imi.values,
        }
    )
    daily = daily.merge(assign[["date", "node"]], on="date", how="inner")

    rows: list[dict[str, float | str]] = []
    for node in ALL_NODES:
        sub = daily[daily["node"] == node]
        wnpmi_values = sub["wnpmi"].to_numpy()
        imi_values = sub["imi"].to_numpy()
        wnpmi_test = stats.ttest_1samp(wnpmi_values, 0.0, nan_policy="omit")
        imi_test = stats.ttest_1samp(imi_values, 0.0, nan_policy="omit")
        rows.append(
            {
                "node": int(node),
                "n": int(len(sub)),
                "wnpmi_p": float(wnpmi_test.pvalue),
                "wnpmi_sig": significance_stars(float(wnpmi_test.pvalue)),
                "imi_p": float(imi_test.pvalue),
                "imi_sig": significance_stars(float(imi_test.pvalue)),
            }
        )
    return pd.DataFrame(rows)


def node_composite_index_table(outdir: Path) -> pd.DataFrame:
    def _from_summary() -> pd.DataFrame:
        summary = pd.read_csv(outdir / "som_node_summary.csv")
        rename_map = {}
        if "node_mean_wnpmi_anom" in summary.columns:
            rename_map["node_mean_wnpmi_anom"] = "composite_wnpmi"
        if "node_mean_imi_anom" in summary.columns:
            rename_map["node_mean_imi_anom"] = "composite_imi"
        if not rename_map:
            raise FileNotFoundError(f"No node composite index source found in {outdir}")
        return summary[["node", *rename_map.keys()]].rename(columns=rename_map)

    if (outdir / "u850_anom_node_composites.nc").exists():
        ds = xr.open_dataset(outdir / "u850_anom_node_composites.nc")
    elif (outdir / "u850_conv_node_composites.nc").exists():
        ds = xr.open_dataset(outdir / "u850_conv_node_composites.nc")
    else:
        return _from_summary()

    try:
        field = ds["uwnd_anom_comp"] if "uwnd_anom_comp" in ds.data_vars else ds["u850_anom_comp"]
        rows: list[dict[str, float]] = []
        for node in ALL_NODES:
            comp = field.sel(node=node)
            wnpmi = area_mean(comp, WNPMI_BOX_A) - area_mean(comp, WNPMI_BOX_B)
            imi = area_mean(comp, IMI_WEST_BOX) - area_mean(comp, IMI_NORTH_BOX)
            rows.append(
                {
                    "node": int(node),
                    "composite_wnpmi": float(wnpmi.values),
                    "composite_imi": float(imi.values),
                }
            )
        return pd.DataFrame(rows)
    except (IndexError, KeyError, ValueError):
        return _from_summary()


def running_mean_shape_stats(occurrence_rm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for node in ALL_NODES:
        col = f"Node_{node}"
        sub = occurrence_rm_df[["year", col]].dropna()
        years = sub["year"].to_numpy(dtype=float)
        values = sub[col].to_numpy(dtype=float)
        if len(sub) < 5:
            rows.append(
                {
                    "node": int(node),
                    "trend_p": np.nan,
                    "trend_sig": "",
                    "trend_slope": np.nan,
                    "cycle_p": np.nan,
                    "cycle_sig": "",
                    "cycle_vertex_year": np.nan,
                }
            )
            continue

        trend = linear_trend_stats(years, values)
        cycle = quadratic_cycle_stats(years, values)
        cycle_sig = (
            np.isfinite(cycle["quad_p"])
            and cycle["quad_p"] < 0.05
            and np.isfinite(cycle["vertex_year"])
            and years.min() <= cycle["vertex_year"] <= years.max()
        )
        rows.append(
            {
                "node": int(node),
                "trend_p": trend["p"],
                "trend_sig": significance_stars(trend["p"]),
                "trend_slope": trend["slope"],
                "cycle_p": cycle["quad_p"],
                "cycle_sig": "*" if cycle_sig else "",
                "cycle_vertex_year": cycle["vertex_year"],
            }
        )
    return pd.DataFrame(rows)


def plot_occurrence_timeseries_with_indices(
    occurrence_df: pd.DataFrame,
    occurrence_rm_df: pd.DataFrame,
    node_index_df: pd.DataFrame,
    out_fig: Path,
) -> None:
    node_cols = [f"Node_{node}" for node in ALL_NODES]
    years = occurrence_df["year"].to_numpy()
    occ_max = float(np.nanmax(occurrence_df[node_cols].to_numpy()))
    metric_table = node_index_df.set_index("node")

    def _extreme_nodes(column: str) -> tuple[set[int], set[int]]:
        values = metric_table[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return set(), set()
        vmin = float(finite.min())
        vmax = float(finite.max())
        node_values = metric_table[column]
        min_nodes = {int(node) for node, value in node_values.items() if np.isfinite(value) and np.isclose(value, vmin)}
        max_nodes = {int(node) for node, value in node_values.items() if np.isfinite(value) and np.isclose(value, vmax)}
        return min_nodes, max_nodes

    wnpmi_min_nodes, wnpmi_max_nodes = _extreme_nodes("composite_wnpmi")
    imi_min_nodes, imi_max_nodes = _extreme_nodes("composite_imi")

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.4))
    fig.subplots_adjust(
        left=0.055,
        right=0.955,
        top=0.905,
        bottom=0.135,
        hspace=0.30,
        wspace=0.22,
    )

    for node, ax in zip(ALL_NODES, axes.flat):
        col = f"Node_{node}"
        ax.bar(
            years,
            occurrence_df[col].to_numpy(),
            width=0.78,
            color="#dbeafe",
            edgecolor="#93c5fd",
            linewidth=0.5,
        )
        ax.plot(
            years,
            occurrence_rm_df[col].to_numpy(),
            color="#111827",
            linewidth=2.2,
        )
        ax.set_title(f"Node {node}", fontsize=11)
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_ylim(0.0, occ_max * 1.15)
        ax.grid(axis="y", linewidth=0.5, alpha=0.30)

        metrics = metric_table.loc[node]
        rm_sub = occurrence_rm_df[["year", col]].dropna()
        rm_years = rm_sub["year"].to_numpy(dtype=float)
        rm_values = rm_sub[col].to_numpy(dtype=float)

        trend = linear_trend_stats(rm_years, rm_values)
        if metrics["trend_sig"] == "*":
            ax.plot(
                rm_years,
                trend["intercept"] + trend["slope"] * rm_years,
                color="#059669",
                linestyle="--",
                linewidth=1.5,
            )

        cycle = quadratic_cycle_stats(rm_years, rm_values)
        if metrics["cycle_sig"] == "*":
            x_centered = rm_years - rm_years.mean()
            cycle_fit = cycle["b0"] + cycle["b1"] * x_centered + cycle["b2"] * x_centered**2
            ax.plot(
                rm_years,
                cycle_fit,
                color="#6b7280",
                linestyle=":",
                linewidth=1.8,
            )

        metric_lines: list[tuple[str, str]] = []
        if node in wnpmi_max_nodes:
            metric_lines.append(("#dc2626", f"WNPMI max {metrics['composite_wnpmi']:+.2f}{metrics['wnpmi_sig']}"))
        elif node in wnpmi_min_nodes:
            metric_lines.append(("#dc2626", f"WNPMI min {metrics['composite_wnpmi']:+.2f}{metrics['wnpmi_sig']}"))
        if node in imi_max_nodes:
            metric_lines.append(("#2563eb", f"IMI max {metrics['composite_imi']:+.2f}{metrics['imi_sig']}"))
        elif node in imi_min_nodes:
            metric_lines.append(("#2563eb", f"IMI min {metrics['composite_imi']:+.2f}{metrics['imi_sig']}"))

        for idx, (color, label) in enumerate(metric_lines):
            ax.text(
                0.97,
                0.95 - 0.09 * idx,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.8,
                color=color,
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
            )
        note_lines = []
        if metrics["trend_sig"] == "*":
            note_lines.append("Trend*")
        if metrics["cycle_sig"] == "*":
            note_lines.append("Cycle*")
        if note_lines:
            ax.text(
                0.03,
                0.95,
                "\n".join(note_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                color="#374151",
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
            )

        if node in (1, 4, 7):
            ax.set_ylabel("Occurrence frequency")
        else:
            ax.set_ylabel("")

        if node <= 6:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year")

    legend_handles = [
        Patch(facecolor="#dbeafe", edgecolor="#93c5fd", label="Annual occurrence"),
        Line2D([0], [0], color="#111827", linewidth=2.2, label="10-year centered RM"),
        Line2D([0], [0], color="#059669", linewidth=1.5, linestyle="--", label="Trend*"),
        Line2D([0], [0], color="#6b7280", linewidth=1.8, linestyle=":", label="Cycle*"),
        Line2D([0], [0], color="none", label="*: p<0.05"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.6,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_occurrence_bundle(repo_root: Path, outdir: Path) -> None:
    occurrence_csv = outdir / "som_occurrence_frequency_by_year.csv"
    occurrence_df = pd.read_csv(occurrence_csv)

    node_cols = [f"Node_{node}" for node in ALL_NODES]
    occurrence_rm_df = centered_running_mean(occurrence_df, node_cols, window=WINDOW)
    node_index_df = node_composite_index_table(outdir).merge(
        daily_node_index_stats(repo_root, outdir),
        on="node",
        how="left",
    ).merge(
        running_mean_shape_stats(occurrence_rm_df),
        on="node",
        how="left",
    )

    occurrence_rm_df.to_csv(outdir / "som_occurrence_frequency_10yr_running_mean.csv", index=False)
    node_index_df.to_csv(outdir / "som_node_composite_indices.csv", index=False)

    plot_occurrence_timeseries_with_indices(
        occurrence_df,
        occurrence_rm_df,
        node_index_df,
        outdir / "som_occurrence_frequency_timeseries.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SOM occurrence timeseries with WNPMI and IMI overlays.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["som_u850", "som_u850_olr"],
        help="Output directories under outputs/ to refresh.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd())
    for target in args.targets:
        render_occurrence_bundle(repo_root, repo_root / "outputs" / target)


if __name__ == "__main__":
    main()
