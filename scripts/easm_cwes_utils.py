from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.lines import Line2D
from scipy import stats


DOMAIN_LAT = (0.0, 60.0)
DOMAIN_LON = (100.0, 180.0)
PLOT_EXTENT = [20.0, 260.0, -20.0, 60.0]
EARLY_PERIOD = (1991, 2006)
LATE_PERIOD = (2007, 2023)

WNPMI_BOX_A = {"lat": (5.0, 15.0), "lon": (100.0, 130.0)}
WNPMI_BOX_B = {"lat": (20.0, 30.0), "lon": (110.0, 140.0)}
IMI_WEST_BOX = {"lat": (5.0, 15.0), "lon": (40.0, 80.0)}
IMI_NORTH_BOX = {"lat": (20.0, 30.0), "lon": (70.0, 90.0)}
WNP_CONV_BOX = {"lat": (10.0, 20.0), "lon": (115.0, 140.0)}
WNP_CWES_BOX = {"lat": (10.0, 20.0), "lon": (145.0, 180.0)}
NIO_BOX = {"lat": (5.0, 25.0), "lon": (40.0, 100.0)}
IOBW_BOX = {"lat": (-20.0, 20.0), "lon": (40.0, 100.0)}
NINO34_BOX = {"lat": (-5.0, 5.0), "lon": (190.0, 240.0)}
ALL_NODES = np.arange(1, 10)
INDEX_LABELS = {"wnpmi": "WNPMI", "imi": "IMI"}


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "scripts").exists()
            and (candidate / "notebooks").exists()
            and (candidate / "data").exists()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate the repository root. Start from this repository or one of its subdirectories."
    )


def _sorted_lon(da: xr.DataArray) -> xr.DataArray:
    if float(da["lon"][0]) > float(da["lon"][-1]):
        da = da.sortby("lon")
    return da


def _sorted_lat(da: xr.DataArray) -> xr.DataArray:
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.sortby("lat")
    return da


def open_field(path: Path | str, varname: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    da = ds[varname]

    rename_dict = {}
    if "latitude" in da.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)

    if "zlev" in da.dims and da.sizes["zlev"] == 1:
        da = da.squeeze("zlev", drop=True)

    da = da.squeeze(drop=True)
    da = da.assign_coords(time=pd.to_datetime(da["time"].values).normalize())
    da = _sorted_lat(_sorted_lon(da))

    dim_order = [dim for dim in ("time", "lat", "lon") if dim in da.dims]
    if tuple(da.dims) != tuple(dim_order):
        da = da.transpose(*dim_order)

    return da


def subset_box(da: xr.DataArray, box: dict[str, tuple[float, float]]) -> xr.DataArray:
    lat_min, lat_max = box["lat"]
    lon_min, lon_max = box["lon"]

    lat_slice = slice(min(lat_min, lat_max), max(lat_min, lat_max))
    if lon_min <= lon_max:
        out = da.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
    else:
        left = da.sel(lat=lat_slice, lon=slice(lon_min, float(da["lon"].max())))
        right = da.sel(lat=lat_slice, lon=slice(float(da["lon"].min()), lon_max))
        out = xr.concat([left, right], dim="lon")

    return _sorted_lat(_sorted_lon(out))


def subset_domain(da: xr.DataArray) -> xr.DataArray:
    return subset_box(
        da,
        {
            "lat": DOMAIN_LAT,
            "lon": DOMAIN_LON,
        },
    )


def align_on_common_time(*arrays: xr.DataArray) -> list[xr.DataArray]:
    common_time = pd.Index(arrays[0]["time"].values)
    for da in arrays[1:]:
        common_time = common_time.intersection(pd.Index(da["time"].values))

    common_time = common_time.sort_values()
    return [da.sel(time=common_time) for da in arrays]


def make_mmdd_anomaly(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    mmdd = xr.DataArray(
        pd.to_datetime(da["time"].values).strftime("%m-%d"),
        coords={"time": da["time"]},
        dims=("time",),
        name="mmdd",
    )
    clim = da.groupby(mmdd).mean("time")
    anom = da.groupby(mmdd) - clim
    return anom, clim


def area_mean(da: xr.DataArray, box: dict[str, tuple[float, float]]) -> xr.DataArray:
    sub = subset_box(da, box)
    weights = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(weights).mean(("lat", "lon"))


def compute_daily_index_table(
    uwnd_raw: xr.DataArray,
    uwnd_anom: xr.DataArray,
    conv: xr.DataArray,
    sst_anom: xr.DataArray,
) -> pd.DataFrame:
    uwnd_raw, uwnd_anom, conv, sst_anom = align_on_common_time(uwnd_raw, uwnd_anom, conv, sst_anom)
    dates = pd.to_datetime(uwnd_raw["time"].values).normalize()

    wnpmi_raw = area_mean(uwnd_raw, WNPMI_BOX_A) - area_mean(uwnd_raw, WNPMI_BOX_B)
    wnpmi_anom = area_mean(uwnd_anom, WNPMI_BOX_A) - area_mean(uwnd_anom, WNPMI_BOX_B)
    imi_raw = area_mean(uwnd_raw, IMI_WEST_BOX) - area_mean(uwnd_raw, IMI_NORTH_BOX)
    imi_anom = area_mean(uwnd_anom, IMI_WEST_BOX) - area_mean(uwnd_anom, IMI_NORTH_BOX)

    df = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "wnpmi_raw": wnpmi_raw.values,
            "wnpmi_anom": wnpmi_anom.values,
            "imi_raw": imi_raw.values,
            "imi_anom": imi_anom.values,
            "wnp_conv": area_mean(conv, WNP_CONV_BOX).values,
            "wnp_cwes_sst": area_mean(sst_anom, WNP_CWES_BOX).values,
            "nio_sst": area_mean(sst_anom, NIO_BOX).values,
            "iobw_sst": area_mean(sst_anom, IOBW_BOX).values,
            "nino34": area_mean(sst_anom, NINO34_BOX).values,
        }
    )

    return df


def compute_annual_index_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    annual = (
        daily_df.groupby("year", as_index=False)
        .agg(
            wnpmi_jja=("wnpmi_raw", "mean"),
            wnpmi_jja_anom=("wnpmi_anom", "mean"),
            imi_jja=("imi_raw", "mean"),
            imi_jja_anom=("imi_anom", "mean"),
            wnp_conv_jja=("wnp_conv", "mean"),
            wnp_cwes_sst_jja=("wnp_cwes_sst", "mean"),
            nio_sst_jja=("nio_sst", "mean"),
            iobw_sst_jja=("iobw_sst", "mean"),
            nino34_jja=("nino34", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    return annual


def top_nodes_from_daily_with_node(daily_with_node: pd.DataFrame, index_col: str) -> list[int]:
    ranked = daily_with_node.groupby("node")[index_col].mean().sort_values(ascending=False)
    return [int(node) for node in ranked.index[:3]]


def merge_top3_occurrence(
    annual_df: pd.DataFrame,
    occurrence_df: pd.DataFrame,
    top3_nodes: list[int],
    out_col: str,
) -> pd.DataFrame:
    top3_cols = [f"Node_{node}" for node in top3_nodes]
    top3_occ = occurrence_df[["year", *top3_cols]].copy()
    top3_occ[out_col] = top3_occ[top3_cols].sum(axis=1)
    return annual_df.merge(top3_occ[["year", out_col]], on="year", how="left")


def make_relationship_row(
    relationship: str,
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    control: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    row = {
        "relationship": relationship,
        **pair_stats(x, y),
    }
    if control is None:
        row.update(
            {
                "partial_r": np.nan,
                "partial_p": np.nan,
                "partial_slope": np.nan,
                "partial_n": np.nan,
            }
        )
        return row

    partial = pair_stats(x, y, control)
    row.update({f"partial_{key}": value for key, value in partial.items()})
    return row


def node_composite_dataset(da: xr.DataArray, assign_df: pd.DataFrame, prefix: str) -> xr.Dataset:
    assign_df = assign_df.copy()
    assign_df["date"] = pd.to_datetime(assign_df["date"]).dt.normalize()
    assign_df = assign_df[assign_df["node"].between(1, 9)].sort_values("date").drop_duplicates("date")

    common_time = np.intersect1d(da["time"].values, assign_df["date"].values)
    if len(common_time) == 0:
        raise ValueError(f"No common dates found for composite dataset: {prefix}")

    selected = da.sel(time=common_time)
    node_map = assign_df.set_index("date")["node"].reindex(pd.to_datetime(common_time))
    valid = ~node_map.isna().to_numpy()
    selected = selected.isel(time=valid)
    node_values = node_map.to_numpy()[valid].astype(int)

    if selected.sizes["time"] == 0:
        raise ValueError(f"No valid time samples remain for composite dataset: {prefix}")

    selected = selected.assign_coords(node=("time", node_values))
    comp = selected.groupby("node").mean("time").reindex(node=ALL_NODES)
    std = selected.groupby("node").std("time").reindex(node=ALL_NODES)
    n = selected.groupby("node").count("time").reindex(node=ALL_NODES)
    se = std / np.sqrt(n)
    sig2 = (np.abs(comp) >= 2.0 * se).fillna(False)

    ds = xr.Dataset(
        {
            f"{prefix}_comp": comp,
            f"{prefix}_std": std,
            f"{prefix}_n": n,
            f"{prefix}_se": se,
            f"{prefix}_sig2": sig2.astype(np.int8),
        }
    )
    ds = ds.assign_coords(node=ALL_NODES)
    return ds


def _sym_limit(data: xr.DataArray) -> float:
    max_abs = np.nanmax(np.abs(data.values))
    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0
    return float(max_abs)


def _apply_sig2_hatching(ax, lons, lats, sig2: xr.DataArray) -> None:
    mask = sig2.astype(bool).astype(int)
    ax.contourf(
        lons,
        lats,
        mask,
        levels=[0.5, 1.5],
        hatches=["...."],
        colors="none",
        transform=ccrs.PlateCarree(),
    )


def add_lonlat_gridlines(
    ax,
    extent: list[float] | tuple[float, float, float, float],
    x_step: int = 40,
    y_step: int = 20,
    trim_edge_xlabels: bool = False,
    xlocs: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ylocs: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabels: list[str] | tuple[str, ...] | None = None,
) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    if xlocs is None:
        x_start = int(np.ceil(lon_min / x_step) * x_step)
        xlocs = np.arange(x_start, lon_max + 0.1, x_step)
    else:
        xlocs = np.asarray(xlocs, dtype=float)
    if ylocs is None:
        y_start = int(np.ceil(lat_min / y_step) * y_step)
        ylocs = np.arange(y_start, lat_max + 0.1, y_step)
    else:
        ylocs = np.asarray(ylocs, dtype=float)
    if trim_edge_xlabels and xlocs.size > 1:
        if np.isclose(xlocs[0], lon_min):
            xlocs = xlocs[1:]
        if xlocs.size > 1 and np.isclose(xlocs[-1], lon_max):
            xlocs = xlocs[:-1]
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        color="#6b7280",
        alpha=0.45,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    ax.set_xticks(xlocs, crs=ccrs.PlateCarree())
    ax.set_yticks(ylocs, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_locator(mticker.FixedLocator(xlocs))
    ax.yaxis.set_major_locator(mticker.FixedLocator(ylocs))
    if xlabels is None:
        ax.xaxis.set_major_formatter(
            LongitudeFormatter(direction_label=True, zero_direction_label=False, degree_symbol="°")
        )
    else:
        ax.set_xticklabels(list(xlabels))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1.2, length=2.5, colors="#374151")


def plot_u850_conv_node_panels(
    comp_ds: xr.Dataset,
    counts: pd.Series,
    out_fig: Path,
) -> None:
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.8, 8.6),
        layout="constrained",
        subplot_kw={"projection": proj},
    )

    conv_vlim = _sym_limit(comp_ds["conv_comp"])
    u_vlim = _sym_limit(comp_ds["u850_anom_comp"])
    conv_levels = np.linspace(-conv_vlim, conv_vlim, 17)
    u_levels = np.linspace(-u_vlim, u_vlim, 9)

    pcm = None
    lon = comp_ds["lon"]
    lat = comp_ds["lat"]

    panel_extent = [100, 180, 0, 60]
    for node, ax in zip(ALL_NODES, axes.flat):
        ax.set_extent(panel_extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.7)
        add_lonlat_gridlines(ax, panel_extent)

        conv = comp_ds["conv_comp"].sel(node=node)
        u850 = comp_ds["u850_anom_comp"].sel(node=node)
        conv_sig2 = comp_ds["conv_sig2"].sel(node=node)
        node_n = int(counts.get(node, 0))

        pcm = ax.contourf(
            lon,
            lat,
            conv,
            levels=conv_levels,
            cmap="RdBu_r",
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        _apply_sig2_hatching(ax, lon, lat, conv_sig2)

        contour_kwargs = {
            "levels": [level for level in u_levels if abs(level) > 1e-12],
            "colors": "k",
            "linewidths": 0.8,
            "transform": ccrs.PlateCarree(),
        }
        cs = ax.contour(lon, lat, u850, **contour_kwargs)
        ax.clabel(cs, fmt="%.1f", inline=True, fontsize=8)
        ax.contour(
            lon,
            lat,
            u850,
            levels=[0.0],
            colors="k",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(f"Node {node} (n={node_n})", fontsize=11)

    cbar = fig.colorbar(pcm, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Convection proxy (-OLR, W m-2)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("3x3 SOM node composites: convection shading and u850 contours", fontsize=16)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_sst_node_panels(
    sst_ds: xr.Dataset,
    counts: pd.Series,
    out_fig: Path,
) -> None:
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.3, 7.4),
        layout="constrained",
        subplot_kw={"projection": proj},
    )

    vlim = _sym_limit(sst_ds["sst_anom_comp"])
    levels = np.linspace(-vlim, vlim, 17)
    pcm = None

    for node, ax in zip(ALL_NODES, axes.flat):
        ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.7)
        add_lonlat_gridlines(ax, PLOT_EXTENT)

        sst = sst_ds["sst_anom_comp"].sel(node=node)
        sig2 = sst_ds["sst_anom_sig2"].sel(node=node)
        node_n = int(counts.get(node, 0))

        pcm = ax.contourf(
            sst_ds["lon"],
            sst_ds["lat"],
            sst,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        _apply_sig2_hatching(ax, sst_ds["lon"], sst_ds["lat"], sig2)
        ax.set_title(f"Node {node} (n={node_n})", fontsize=11)

    cbar = fig.colorbar(pcm, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("SST anomaly (degC)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("SST anomaly composites by u850 + OLR SOM node", fontsize=16)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def long_table_to_dataarray(df: pd.DataFrame, value_col: str) -> xr.DataArray:
    nodes = np.sort(df["node"].unique())
    lat = np.sort(df["lat"].unique())
    lon = np.sort(df["lon"].unique())

    node_index = {value: i for i, value in enumerate(nodes)}
    lat_index = {value: i for i, value in enumerate(lat)}
    lon_index = {value: i for i, value in enumerate(lon)}

    values = np.full((len(nodes), len(lat), len(lon)), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        values[
            node_index[row.node],
            lat_index[row.lat],
            lon_index[row.lon],
        ] = getattr(row, value_col)

    return xr.DataArray(
        values,
        coords={"node": nodes, "lat": lat, "lon": lon},
        dims=("node", "lat", "lon"),
        name=value_col,
    )


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]

    design = np.column_stack([np.ones(y.shape[0]), controls])
    pinv = np.linalg.pinv(design)
    if y.ndim == 1:
        beta = pinv @ y
        return y - design @ beta

    beta = pinv @ y
    return y - design @ beta


def pair_stats(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, control: pd.Series | np.ndarray | None = None) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if control is not None:
        control = np.asarray(control, dtype=float)
        mask &= np.isfinite(control)
        control = control[mask]

    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return {"n": float(x.size), "r": np.nan, "p": np.nan, "slope": np.nan}

    if control is not None:
        x = _residualize(x, control)
        y = _residualize(y, control)
        df = x.size - 3
    else:
        df = x.size - 2

    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if denom == 0:
        return {"n": float(x.size), "r": np.nan, "p": np.nan, "slope": np.nan}

    r = float(np.clip(np.sum(xm * ym) / denom, -1.0, 1.0))
    slope = float(np.sum(xm * ym) / np.sum(xm**2))
    if df <= 0 or abs(r) >= 1.0:
        pval = 0.0 if np.isfinite(r) else np.nan
    else:
        tval = r * np.sqrt(df / max(1e-12, 1.0 - r**2))
        pval = float(2.0 * stats.t.sf(abs(tval), df))

    return {"n": float(x.size), "r": r, "p": pval, "slope": slope}


def benjamini_hochberg_mask(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    valid = np.isfinite(pvals)
    if not np.any(valid):
        return np.zeros_like(pvals, dtype=bool)

    flat = pvals[valid]
    order = np.argsort(flat)
    ranked = flat[order]
    thresholds = alpha * np.arange(1, ranked.size + 1) / ranked.size
    keep = ranked <= thresholds
    if not np.any(keep):
        return np.zeros_like(pvals, dtype=bool)

    cutoff = ranked[np.where(keep)[0].max()]
    return valid & (pvals <= cutoff)


def regression_map(
    y_field: xr.DataArray,
    x_index: pd.Series | np.ndarray,
    control: pd.Series | np.ndarray | None = None,
) -> xr.Dataset:
    x = np.asarray(x_index, dtype=float)
    if y_field.shape[0] != x.size:
        raise ValueError("The leading dimension of y_field must match the length of x_index.")

    y = y_field.values.reshape(x.size, -1)
    valid = np.all(np.isfinite(y), axis=0)
    if not np.all(np.isfinite(x)):
        raise ValueError("x_index contains missing values.")

    result_r = np.full(y.shape[1], np.nan, dtype=float)
    result_p = np.full(y.shape[1], np.nan, dtype=float)
    result_slope = np.full(y.shape[1], np.nan, dtype=float)

    if control is not None:
        control = np.asarray(control, dtype=float)
        if not np.all(np.isfinite(control)):
            raise ValueError("control contains missing values.")

    if np.any(valid):
        yv = y[:, valid]
        xv = x.copy()

        if control is not None:
            cv = control.copy()
            xv = _residualize(xv, cv)
            yv = _residualize(yv, cv)
            df = x.size - 3
        else:
            df = x.size - 2

        xm = xv - xv.mean()
        ym = yv - yv.mean(axis=0)
        sum_x2 = np.sum(xm**2)
        denom = np.sqrt(sum_x2 * np.sum(ym**2, axis=0))
        cov_valid = np.sum(xm[:, None] * ym, axis=0)
        ok = np.isfinite(denom) & (denom > 0) & np.isfinite(cov_valid)

        r_valid = np.full(yv.shape[1], np.nan, dtype=float)
        slope_valid = np.full(yv.shape[1], np.nan, dtype=float)
        p_valid = np.full(yv.shape[1], np.nan, dtype=float)

        r_valid[ok] = np.clip(cov_valid[ok] / denom[ok], -1.0, 1.0)
        slope_valid[ok] = cov_valid[ok] / sum_x2
        t_valid = r_valid[ok] * np.sqrt(df / np.maximum(1e-12, 1.0 - r_valid[ok] ** 2))
        p_valid[ok] = 2.0 * stats.t.sf(np.abs(t_valid), df)

        result_r[valid] = r_valid
        result_p[valid] = p_valid
        result_slope[valid] = slope_valid

    shape_2d = y_field.shape[1:]
    coords = {"lat": y_field["lat"], "lon": y_field["lon"]}
    p_2d = result_p.reshape(shape_2d)
    ds = xr.Dataset(
        {
            "corr": xr.DataArray(result_r.reshape(shape_2d), coords=coords, dims=("lat", "lon")),
            "pval": xr.DataArray(p_2d, coords=coords, dims=("lat", "lon")),
            "slope": xr.DataArray(result_slope.reshape(shape_2d), coords=coords, dims=("lat", "lon")),
            "fdr_sig": xr.DataArray(
                benjamini_hochberg_mask(p_2d).astype(np.int8),
                coords=coords,
                dims=("lat", "lon"),
            ),
        }
    )
    return ds


def _scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    raw_stats: dict[str, float],
    partial_stats: dict[str, float] | None = None,
) -> None:
    ax.scatter(x, y, s=35, color="tab:blue", alpha=0.85)
    if np.isfinite(raw_stats["slope"]):
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = np.nanmean(y) + raw_stats["slope"] * (x_line - np.nanmean(x))
        ax.plot(x_line, y_line, color="black", linewidth=1.0)

    note = f"r={raw_stats['r']:.2f}, p={raw_stats['p']:.3f}"
    if partial_stats is not None:
        note += f"\npartial r={partial_stats['r']:.2f}, p={partial_stats['p']:.3f}"

    ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.axhline(0.0, color="0.65", linewidth=0.8)
    ax.axvline(0.0, color="0.65", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def plot_daily_scatter(
    daily_df: pd.DataFrame,
    x_col: str,
    out_fig: Path,
    stats_row: dict[str, float],
    *,
    xlabel: str,
    title: str,
    period_stats: list[dict[str, object]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
    if period_stats:
        all_x = daily_df[x_col].to_numpy()
        all_y = daily_df["wnp_conv"].to_numpy()
        ax.scatter(all_x, all_y, s=12, color="0.82", alpha=0.45, zorder=1)
        if np.isfinite(stats_row["slope"]):
            x_line = np.linspace(np.nanmin(all_x), np.nanmax(all_x), 100)
            y_line = np.nanmean(all_y) + stats_row["slope"] * (x_line - np.nanmean(all_x))
            ax.plot(x_line, y_line, color="black", linewidth=1.1, zorder=3)

        note_lines = [f"All: r={stats_row['r']:.2f}, p={stats_row['p']:.3f}"]
        legend_handles = []
        for spec in period_stats:
            mask = np.asarray(spec["mask"], dtype=bool)
            color = str(spec["color"])
            label = str(spec["label"])
            period_row = spec["stats"]
            x = daily_df.loc[mask, x_col].to_numpy()
            y = daily_df.loc[mask, "wnp_conv"].to_numpy()
            ax.scatter(x, y, s=18, color=color, alpha=0.75, zorder=2)
            if np.isfinite(period_row["slope"]) and np.isfinite(x).any():
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                y_line = np.nanmean(y) + period_row["slope"] * (x_line - np.nanmean(x))
                ax.plot(x_line, y_line, color=color, linewidth=1.3, zorder=4)
            note_lines.append(f"{label}: r={period_row['r']:.2f}, p={period_row['p']:.3f}")
            legend_handles.append(Line2D([0], [0], marker="o", linestyle="-", color=color, label=label))

        ax.text(0.03, 0.97, "\n".join(note_lines), transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.axvline(0.0, color="0.65", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Daily WNP convection index (-OLR, W m-2)")
        ax.set_title(title, fontsize=10)
    else:
        _scatter_panel(
            ax=ax,
            x=daily_df[x_col].to_numpy(),
            y=daily_df["wnp_conv"].to_numpy(),
            xlabel=xlabel,
            ylabel="Daily WNP convection index (-OLR, W m-2)",
            title=title,
            raw_stats=stats_row,
            partial_stats=None,
        )
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_annual_scatter(
    annual_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    top3_nodes: dict[str, list[int]],
    out_fig: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")

    def _row(name: str) -> pd.Series:
        return stats_df.set_index("relationship").loc[name]

    scatter_specs = [
        (
            "annual_wnpmi_vs_wnp_cwes_sst",
            annual_df["wnpmi_jja"].to_numpy(),
            annual_df["wnp_cwes_sst_jja"].to_numpy(),
            "WNPMI JJA index (m/s)",
            "WNP local CWES SST (degC)",
            "WNPMI vs WNP local CWES SST",
        ),
        (
            "annual_wnpmi_vs_nio_sst",
            annual_df["wnpmi_jja"].to_numpy(),
            annual_df["nio_sst_jja"].to_numpy(),
            "WNPMI JJA index (m/s)",
            "Northern IO SST (degC)",
            "WNPMI vs northern Indian Ocean SST",
        ),
        (
            "annual_top3_wnpmi_occ_vs_wnp_cwes_sst",
            annual_df["top3_wnpmi_occ"].to_numpy(),
            annual_df["wnp_cwes_sst_jja"].to_numpy(),
            "Top-3 WNPMI node occurrence",
            "WNP local CWES SST (degC)",
            f"Top-3 WNPMI nodes {top3_nodes['wnpmi']} vs WNP local CWES SST",
        ),
        (
            "annual_top3_wnpmi_occ_vs_nio_sst",
            annual_df["top3_wnpmi_occ"].to_numpy(),
            annual_df["nio_sst_jja"].to_numpy(),
            "Top-3 WNPMI node occurrence",
            "Northern IO SST (degC)",
            f"Top-3 WNPMI nodes {top3_nodes['wnpmi']} vs northern IO SST",
        ),
        (
            "annual_imi_vs_iobw_sst",
            annual_df["imi_jja"].to_numpy(),
            annual_df["iobw_sst_jja"].to_numpy(),
            "IMI JJA index (m/s)",
            "IOBW SST (degC)",
            "IMI vs Indian Ocean basinwide SST",
        ),
        (
            "annual_imi_vs_nio_sst",
            annual_df["imi_jja"].to_numpy(),
            annual_df["nio_sst_jja"].to_numpy(),
            "IMI JJA index (m/s)",
            "Northern IO SST (degC)",
            "IMI vs northern Indian Ocean SST",
        ),
        (
            "annual_top3_imi_occ_vs_iobw_sst",
            annual_df["top3_imi_occ"].to_numpy(),
            annual_df["iobw_sst_jja"].to_numpy(),
            "Top-3 IMI node occurrence",
            "IOBW SST (degC)",
            f"Top-3 IMI nodes {top3_nodes['imi']} vs IOBW SST",
        ),
        (
            "annual_top3_imi_occ_vs_nio_sst",
            annual_df["top3_imi_occ"].to_numpy(),
            annual_df["nio_sst_jja"].to_numpy(),
            "Top-3 IMI node occurrence",
            "Northern IO SST (degC)",
            f"Top-3 IMI nodes {top3_nodes['imi']} vs northern IO SST",
        ),
    ]

    for ax, (name, x, y, xlabel, ylabel, title) in zip(axes.flat, scatter_specs):
        row = _row(name)
        raw_stats = {"r": row["r"], "p": row["p"], "slope": row["slope"], "n": row["n"]}
        partial_stats = {
            "r": row["partial_r"],
            "p": row["partial_p"],
            "slope": row["partial_slope"],
            "n": row["n"],
        }
        _scatter_panel(ax, x, y, xlabel, ylabel, title, raw_stats, partial_stats)

    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_regression_maps(map_ds: xr.Dataset, out_fig: Path) -> None:
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        layout="constrained",
        subplot_kw={"projection": proj},
    )

    panels = [
        ("corr_raw", "Raw correlation", "RdBu_r", (-1.0, 1.0)),
        ("slope_raw", "Raw regression slope", "RdBu_r", None),
        ("corr_partial", "Partial correlation | Niño3.4", "RdBu_r", (-1.0, 1.0)),
        ("slope_partial", "Partial regression slope | Niño3.4", "RdBu_r", None),
    ]

    for row_idx, index_key in enumerate(("wnpmi", "imi")):
        for col_idx, (name, title, cmap, fixed_range) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            field = map_ds[f"{index_key}_{name}"]
            sig = map_ds[f"{index_key}_{name}_sig"]

            if fixed_range is None:
                vmax = _sym_limit(field)
                vmin = -vmax
            else:
                vmin, vmax = fixed_range

            ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.7)
            pcm = ax.contourf(
                field["lon"],
                field["lat"],
                field,
                levels=np.linspace(vmin, vmax, 17),
                cmap=cmap,
                extend="both",
                transform=ccrs.PlateCarree(),
            )
            _apply_sig2_hatching(ax, field["lon"], field["lat"], sig)
            ax.set_title(f"{INDEX_LABELS[index_key]}: {title}", fontsize=10)
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.82, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Annual JJA SST maps regressed onto WNPMI and IMI", fontsize=14)
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def interdecadal_node_table(
    occurrence_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    node_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = occurrence_df.merge(
        annual_df[["year", "nio_sst_jja", "wnp_cwes_sst_jja", "nino34_jja", "wnpmi_jja", "imi_jja"]],
        on="year",
        how="inner",
    )
    early_mask = merged["year"].between(*EARLY_PERIOD)
    late_mask = merged["year"].between(*LATE_PERIOD)

    rows: list[dict[str, float]] = []
    for node in ALL_NODES:
        col = f"Node_{node}"
        series = merged[col]
        trend = pair_stats(merged["year"], series)
        nio = pair_stats(series, merged["nio_sst_jja"])
        nio_partial = pair_stats(series, merged["nio_sst_jja"], merged["nino34_jja"])
        cwes = pair_stats(series, merged["wnp_cwes_sst_jja"])
        cwes_partial = pair_stats(series, merged["wnp_cwes_sst_jja"], merged["nino34_jja"])
        wnpmi = pair_stats(series, merged["wnpmi_jja"])
        imi = pair_stats(series, merged["imi_jja"])

        early_values = series[early_mask].to_numpy()
        late_values = series[late_mask].to_numpy()
        ttest = stats.ttest_ind(late_values, early_values, equal_var=False, nan_policy="omit")

        rows.append(
            {
                "node": node,
                "early_mean_occ": float(np.nanmean(early_values)),
                "late_mean_occ": float(np.nanmean(late_values)),
                "late_minus_early": float(np.nanmean(late_values) - np.nanmean(early_values)),
                "late_vs_early_p": float(ttest.pvalue),
                "trend_per_decade": float(trend["slope"] * 10.0),
                "trend_p": float(trend["p"]),
                "nio_r": float(nio["r"]),
                "nio_p": float(nio["p"]),
                "nio_partial_r": float(nio_partial["r"]),
                "nio_partial_p": float(nio_partial["p"]),
                "cwes_r": float(cwes["r"]),
                "cwes_p": float(cwes["p"]),
                "cwes_partial_r": float(cwes_partial["r"]),
                "cwes_partial_p": float(cwes_partial["p"]),
                "wnpmi_r": float(wnpmi["r"]),
                "wnpmi_p": float(wnpmi["p"]),
                "imi_r": float(imi["r"]),
                "imi_p": float(imi["p"]),
            }
        )

    out = pd.DataFrame(rows).merge(node_summary_df, on="node", how="left")
    return out.sort_values("node").reset_index(drop=True)


def plot_interdecadal_occurrence_panels(
    occurrence_df: pd.DataFrame,
    interdecadal_df: pd.DataFrame,
    out_fig: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    years = occurrence_df["year"].to_numpy()

    for node, ax in zip(ALL_NODES, axes.flat):
        row = interdecadal_df.set_index("node").loc[node]
        col = f"Node_{node}"
        values = occurrence_df[col].to_numpy()

        ax.axvspan(EARLY_PERIOD[0] - 0.5, EARLY_PERIOD[1] + 0.5, color="#d9e6f2", alpha=0.6)
        ax.axvspan(LATE_PERIOD[0] - 0.5, LATE_PERIOD[1] + 0.5, color="#f4d6cc", alpha=0.6)
        ax.plot(years, values, color="0.25", linewidth=1.1)
        ax.scatter(years, values, s=20, color="tab:blue", alpha=0.9)
        ax.axhline(row["early_mean_occ"], color="#4c78a8", linestyle="--", linewidth=1.0)
        ax.axhline(row["late_mean_occ"], color="#d62728", linestyle="--", linewidth=1.0)

        if np.isfinite(row["trend_per_decade"]):
            fit = np.polyfit(years, values, 1)
            ax.plot(years, fit[0] * years + fit[1], color="black", linewidth=1.0)

        ax.text(
            0.03,
            0.97,
            f"Δ={row['late_minus_early']:+.3f}\ntrend={row['trend_per_decade']:+.3f}/dec\np={row['trend_p']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.set_title(f"Node {node}", fontsize=10)
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_ylim(0.0, max(occurrence_df.drop(columns=["year"]).max()) * 1.1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Occurrence")

    fig.suptitle("Inter-decadal change in SOM node occurrence (JJA)", fontsize=14)
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interdecadal_summary(interdecadal_df: pd.DataFrame, out_fig: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    order = interdecadal_df["node"].to_numpy()
    colors = ["#d62728" if delta > 0 else "#1f77b4" for delta in interdecadal_df["late_minus_early"]]
    axes[0].bar(order, interdecadal_df["late_minus_early"], color=colors)
    axes[0].axhline(0.0, color="0.2", linewidth=1.0)
    axes[0].set_title("Late - early occurrence")
    axes[0].set_xlabel("Node")
    axes[0].set_ylabel("Occurrence difference")

    axes[1].bar(order - 0.18, interdecadal_df["nio_partial_r"], width=0.36, label="NIO | Niño3.4", color="#2a9d8f")
    axes[1].bar(order + 0.18, interdecadal_df["cwes_partial_r"], width=0.36, label="WNP CWES | Niño3.4", color="#e76f51")
    axes[1].axhline(0.0, color="0.2", linewidth=1.0)
    axes[1].set_title("Partial correlation with SST indices")
    axes[1].set_xlabel("Node")
    axes[1].set_ylabel("Partial r")
    axes[1].legend(frameon=False)

    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_node_interpretation(repo_root: Path | None = None) -> dict[str, Path]:
    repo_root = find_repo_root(Path.cwd() if repo_root is None else repo_root)
    processed_dir = repo_root / "data" / "processed"
    outdir = repo_root / "outputs" / "som_u850_olr"
    outdir.mkdir(parents=True, exist_ok=True)

    assign_df = pd.read_csv(outdir / "som_daily_assignment.csv")
    assign_df["date"] = pd.to_datetime(assign_df["date"]).dt.normalize()

    uwnd = open_field(processed_dir / "uwnd_z850_jja_1991_2023.nc", "uwnd")
    olr = open_field(processed_dir / "olr_jja_1991_2023.nc", "olr")
    sst = open_field(processed_dir / "oisst_jja_1991_2023_on_uwnd_grid.nc", "anom")
    uwnd, olr, sst = align_on_common_time(uwnd, olr, sst)

    uwnd_anom, _ = make_mmdd_anomaly(uwnd)
    conv = -olr
    conv.name = "conv"

    daily_df = compute_daily_index_table(uwnd, uwnd_anom, conv, sst)
    daily_with_node = daily_df.merge(assign_df[["date", "node"]], on="date", how="inner")
    node_counts = daily_with_node.groupby("node").size().reindex(ALL_NODES, fill_value=0)

    occurrence_df = pd.read_csv(outdir / "som_occurrence_frequency_by_year.csv")
    mean_occurrence = occurrence_df.drop(columns=["year"]).mean(axis=0)
    mean_occurrence.index = [int(name.split("_")[1]) for name in mean_occurrence.index]

    node_summary = (
        daily_with_node.groupby("node", as_index=False)
        .agg(
            node_count=("date", "size"),
            node_mean_wnpmi_anom=("wnpmi_anom", "mean"),
            node_mean_imi_anom=("imi_anom", "mean"),
            node_mean_wnp_conv=("wnp_conv", "mean"),
        )
        .sort_values("node")
    )
    node_summary["mean_annual_occurrence"] = node_summary["node"].map(mean_occurrence)
    node_summary.to_csv(outdir / "som_node_summary.csv", index=False)

    u_codebook_csv = outdir / "som_u850_codebook_maps.csv"
    conv_codebook_csv = outdir / "som_conv_codebook_maps.csv"
    if u_codebook_csv.exists() and conv_codebook_csv.exists():
        u_codebook_df = pd.read_csv(u_codebook_csv)
        conv_codebook_df = pd.read_csv(conv_codebook_csv)
        codebook_ds = xr.Dataset(
            {
                "u850_codebook": long_table_to_dataarray(u_codebook_df, "u850_codebook"),
                "conv_codebook": long_table_to_dataarray(conv_codebook_df, "conv_codebook"),
            }
        )
        codebook_ds.to_netcdf(outdir / "u850_conv_codebook_maps.nc")

    u_domain = subset_domain(uwnd_anom)
    conv_domain = subset_domain(conv)
    u_ds = node_composite_dataset(u_domain, assign_df, "u850_anom")
    conv_ds = node_composite_dataset(conv_domain, assign_df, "conv")
    merged_ds = xr.merge([u_ds, conv_ds])
    merged_ds.to_netcdf(outdir / "u850_conv_node_composites.nc")
    plot_u850_conv_node_panels(merged_ds, node_counts, outdir / "u850_conv_node_composites.png")

    sst_ds = node_composite_dataset(sst, assign_df, "sst_anom")
    sst_ds.to_netcdf(outdir / "sst_node_composites.nc")
    plot_sst_node_panels(sst_ds, node_counts, outdir / "sst_anom_node_composites_sig2.png")

    return {
        "node_summary_csv": outdir / "som_node_summary.csv",
        "codebook_nc": outdir / "u850_conv_codebook_maps.nc",
        "u850_conv_nc": outdir / "u850_conv_node_composites.nc",
        "u850_conv_png": outdir / "u850_conv_node_composites.png",
        "sst_nc": outdir / "sst_node_composites.nc",
        "sst_png": outdir / "sst_anom_node_composites_sig2.png",
    }


def run_easm_cwes_validation(repo_root: Path | None = None) -> dict[str, Path]:
    repo_root = find_repo_root(Path.cwd() if repo_root is None else repo_root)
    processed_dir = repo_root / "data" / "processed"
    outdir = repo_root / "outputs" / "som_u850_olr"
    outdir.mkdir(parents=True, exist_ok=True)

    assign_df = pd.read_csv(outdir / "som_daily_assignment.csv")
    assign_df["date"] = pd.to_datetime(assign_df["date"]).dt.normalize()

    uwnd = open_field(processed_dir / "uwnd_z850_jja_1991_2023.nc", "uwnd")
    olr = open_field(processed_dir / "olr_jja_1991_2023.nc", "olr")
    sst = open_field(processed_dir / "oisst_jja_1991_2023_on_uwnd_grid.nc", "anom")
    uwnd, olr, sst = align_on_common_time(uwnd, olr, sst)

    uwnd_anom, _ = make_mmdd_anomaly(uwnd)
    conv = -olr
    conv.name = "conv"

    daily_df = compute_daily_index_table(uwnd, uwnd_anom, conv, sst)
    daily_df.to_csv(outdir / "easm_cwes_daily_indices.csv", index=False)
    daily_df["period"] = np.where(daily_df["year"].between(*EARLY_PERIOD), "early", "late")

    annual_df = compute_annual_index_table(daily_df)
    annual_df["wnpmi_jja_centered"] = annual_df["wnpmi_jja"] - annual_df["wnpmi_jja"].mean()
    annual_df["imi_jja_centered"] = annual_df["imi_jja"] - annual_df["imi_jja"].mean()

    daily_with_node = daily_df.merge(assign_df[["date", "node"]], on="date", how="inner")
    top3_wnpmi_nodes = top_nodes_from_daily_with_node(daily_with_node, "wnpmi_anom")
    top3_imi_nodes = top_nodes_from_daily_with_node(daily_with_node, "imi_anom")

    occurrence_df = pd.read_csv(outdir / "som_occurrence_frequency_by_year.csv")
    annual_df = merge_top3_occurrence(annual_df, occurrence_df, top3_wnpmi_nodes, "top3_wnpmi_occ")
    annual_df = merge_top3_occurrence(annual_df, occurrence_df, top3_imi_nodes, "top3_imi_occ")
    annual_df.to_csv(outdir / "easm_cwes_annual_indices.csv", index=False)

    stats_rows = [
        make_relationship_row("daily_wnpmianom_vs_wnp_conv", daily_df["wnpmi_anom"], daily_df["wnp_conv"]),
        make_relationship_row("daily_imianom_vs_wnp_conv", daily_df["imi_anom"], daily_df["wnp_conv"]),
        make_relationship_row(
            "daily_imianom_vs_wnp_conv_early",
            daily_df.loc[daily_df["period"] == "early", "imi_anom"],
            daily_df.loc[daily_df["period"] == "early", "wnp_conv"],
        ),
        make_relationship_row(
            "daily_imianom_vs_wnp_conv_late",
            daily_df.loc[daily_df["period"] == "late", "imi_anom"],
            daily_df.loc[daily_df["period"] == "late", "wnp_conv"],
        ),
        make_relationship_row(
            "annual_wnpmi_vs_wnp_cwes_sst",
            annual_df["wnpmi_jja"],
            annual_df["wnp_cwes_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_wnpmi_vs_nio_sst",
            annual_df["wnpmi_jja"],
            annual_df["nio_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_wnpmi_vs_iobw_sst",
            annual_df["wnpmi_jja"],
            annual_df["iobw_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_imi_vs_nio_sst",
            annual_df["imi_jja"],
            annual_df["nio_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_imi_vs_iobw_sst",
            annual_df["imi_jja"],
            annual_df["iobw_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_top3_wnpmi_occ_vs_wnp_cwes_sst",
            annual_df["top3_wnpmi_occ"],
            annual_df["wnp_cwes_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_top3_wnpmi_occ_vs_nio_sst",
            annual_df["top3_wnpmi_occ"],
            annual_df["nio_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_top3_imi_occ_vs_iobw_sst",
            annual_df["top3_imi_occ"],
            annual_df["iobw_sst_jja"],
            annual_df["nino34_jja"],
        ),
        make_relationship_row(
            "annual_top3_imi_occ_vs_nio_sst",
            annual_df["top3_imi_occ"],
            annual_df["nio_sst_jja"],
            annual_df["nino34_jja"],
        ),
    ]

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(outdir / "easm_cwes_scalar_relationships.csv", index=False)

    sst_year = sst.groupby("time.year").mean("time").sel(year=annual_df["year"].values)
    map_ds = xr.Dataset()
    for index_key in ("wnpmi", "imi"):
        raw_map = regression_map(sst_year, annual_df[f"{index_key}_jja"].to_numpy())
        partial_map = regression_map(
            sst_year,
            annual_df[f"{index_key}_jja"].to_numpy(),
            annual_df["nino34_jja"].to_numpy(),
        )
        map_ds[f"{index_key}_corr_raw"] = raw_map["corr"]
        map_ds[f"{index_key}_pval_raw"] = raw_map["pval"]
        map_ds[f"{index_key}_slope_raw"] = raw_map["slope"]
        map_ds[f"{index_key}_corr_raw_sig"] = raw_map["fdr_sig"]
        map_ds[f"{index_key}_corr_partial"] = partial_map["corr"]
        map_ds[f"{index_key}_pval_partial"] = partial_map["pval"]
        map_ds[f"{index_key}_slope_partial"] = partial_map["slope"]
        map_ds[f"{index_key}_corr_partial_sig"] = partial_map["fdr_sig"]
        map_ds[f"{index_key}_slope_raw_sig"] = map_ds[f"{index_key}_corr_raw_sig"]
        map_ds[f"{index_key}_slope_partial_sig"] = map_ds[f"{index_key}_corr_partial_sig"]
    map_ds.to_netcdf(outdir / "monsoon_sst_regression_maps.nc")

    daily_stats = stats_df.set_index("relationship").loc["daily_wnpmianom_vs_wnp_conv"].to_dict()
    daily_imi_stats = stats_df.set_index("relationship").loc["daily_imianom_vs_wnp_conv"].to_dict()
    daily_imi_early_stats = stats_df.set_index("relationship").loc["daily_imianom_vs_wnp_conv_early"].to_dict()
    daily_imi_late_stats = stats_df.set_index("relationship").loc["daily_imianom_vs_wnp_conv_late"].to_dict()
    plot_daily_scatter(
        daily_df,
        "wnpmi_anom",
        outdir / "wnpmi_vs_wnp_convection_daily.png",
        daily_stats,
        xlabel="Daily WNPMI anomaly (m/s)",
        title="Daily WNPMI-WNP convection coupling",
    )
    plot_daily_scatter(
        daily_df,
        "imi_anom",
        outdir / "imi_vs_wnp_convection_daily.png",
        daily_imi_stats,
        xlabel="Daily IMI anomaly (m/s)",
        title="Daily IMI-WNP convection coupling",
        period_stats=[
            {
                "label": "Early (1991-2006)",
                "mask": daily_df["period"] == "early",
                "color": "#2563eb",
                "stats": daily_imi_early_stats,
            },
            {
                "label": "Late (2007-2023)",
                "mask": daily_df["period"] == "late",
                "color": "#dc2626",
                "stats": daily_imi_late_stats,
            },
        ],
    )
    plot_annual_scatter(
        annual_df,
        stats_df,
        {"wnpmi": top3_wnpmi_nodes, "imi": top3_imi_nodes},
        outdir / "easm_cwes_annual_scatter.png",
    )
    plot_regression_maps(map_ds, outdir / "monsoon_sst_regression_maps.png")

    with (outdir / "top3_wnpmi_nodes.txt").open("w", encoding="utf-8") as f:
        f.write(",".join(str(node) for node in top3_wnpmi_nodes) + "\n")
    with (outdir / "top3_imi_nodes.txt").open("w", encoding="utf-8") as f:
        f.write(",".join(str(node) for node in top3_imi_nodes) + "\n")

    return {
        "daily_csv": outdir / "easm_cwes_daily_indices.csv",
        "annual_csv": outdir / "easm_cwes_annual_indices.csv",
        "stats_csv": outdir / "easm_cwes_scalar_relationships.csv",
        "map_nc": outdir / "monsoon_sst_regression_maps.nc",
        "daily_png": outdir / "wnpmi_vs_wnp_convection_daily.png",
        "daily_imi_png": outdir / "imi_vs_wnp_convection_daily.png",
        "annual_png": outdir / "easm_cwes_annual_scatter.png",
        "map_png": outdir / "monsoon_sst_regression_maps.png",
    }


def run_interdecadal_wnp_nio_analysis(repo_root: Path | None = None) -> dict[str, Path]:
    repo_root = find_repo_root(Path.cwd() if repo_root is None else repo_root)
    outdir = repo_root / "outputs" / "som_u850_olr"
    outdir.mkdir(parents=True, exist_ok=True)

    occurrence_df = pd.read_csv(outdir / "som_occurrence_frequency_by_year.csv")
    annual_df = pd.read_csv(outdir / "easm_cwes_annual_indices.csv")
    node_summary_df = pd.read_csv(outdir / "som_node_summary.csv")

    interdecadal_df = interdecadal_node_table(occurrence_df, annual_df, node_summary_df)
    interdecadal_df.to_csv(outdir / "som_interdecadal_node_summary.csv", index=False)

    plot_interdecadal_occurrence_panels(
        occurrence_df,
        interdecadal_df,
        outdir / "som_interdecadal_occurrence_timeseries.png",
    )
    plot_interdecadal_summary(
        interdecadal_df,
        outdir / "som_interdecadal_summary.png",
    )

    return {
        "summary_csv": outdir / "som_interdecadal_node_summary.csv",
        "timeseries_png": outdir / "som_interdecadal_occurrence_timeseries.png",
        "summary_png": outdir / "som_interdecadal_summary.png",
    }
