from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from easm_cwes_utils import ALL_NODES, add_lonlat_gridlines, find_repo_root, open_field


PANEL_EXTENT = [20, 230.0, -20.0, 60.0]
PANEL_YTICKS = [0.0, 20.0, 40.0]
PANEL_XTICKS = [40.0, 80.0, 120.0, 160.0]
PANEL_XTICK_LABELS = ["40°E", "80°E", "120°E", "160°E"]
PANEL_YTICK_LABELS = ["EQ", "20°N", "40°N"]
JJA_MONTHS = (6, 7, 8)
OLR_CLIM_START_YEAR = 1991
OLR_CLIM_END_YEAR = 2020
OLR_END_YEAR = 2023
SST_CLIM_START_YEAR = 1991
SST_CLIM_END_YEAR = 2020
SST_END_YEAR = 2023


def load_assignments(outdir: Path) -> pd.DataFrame:
    df = pd.read_csv(outdir / "som_daily_assignment.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["node"].between(1, 9)].copy()
    return df.sort_values("date").drop_duplicates(subset="date")


def align_field_with_nodes(field: xr.DataArray, assignments: pd.DataFrame) -> xr.DataArray:
    common_time = np.intersect1d(field["time"].values, assignments["date"].values)
    if len(common_time) == 0:
        raise ValueError("No common dates were found between the field and SOM assignments.")

    field_sel = field.sel(time=common_time)
    node_map = assignments.set_index("date")["node"].reindex(pd.to_datetime(common_time))
    valid = ~node_map.isna().to_numpy()
    field_sel = field_sel.isel(time=valid)
    node_vals = node_map.to_numpy()[valid].astype(int)
    if field_sel.sizes["time"] == 0:
        raise ValueError("Aligned field has zero time samples after date matching.")
    return field_sel.assign_coords(node=("time", node_vals))


def composite_by_node(field: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    comp = field.groupby("node").mean("time").reindex(node=ALL_NODES)
    std = field.groupby("node").std("time").reindex(node=ALL_NODES)
    n = field.groupby("node").count("time").reindex(node=ALL_NODES)
    se = std / np.sqrt(n)
    sig2 = (np.abs(comp) >= 2.0 * se).fillna(False)
    return comp, std, n, sig2.astype(np.int8)


def subset_panel_domain(field: xr.DataArray, extent: list[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = extent
    lat_sel = field.sel(lat=slice(lat_min, lat_max))
    if lon_min >= 0:
        return lat_sel.sel(lon=slice(lon_min, lon_max))

    west = lat_sel.sel(lon=slice(360.0 + lon_min, 360.0)).assign_coords(
        lon=lambda arr: arr["lon"] - 360.0
    )
    east = lat_sel.sel(lon=slice(0.0, lon_max))
    merged = xr.concat([west, east], dim="lon")
    return merged.sortby("lon")


def oisst_jja_file_list(raw_dir: Path, start_year: int, end_year: int) -> list[Path]:
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        for month in JJA_MONTHS:
            files.extend(sorted((raw_dir / f"{year}{month:02d}").glob("*.nc")))
    if not files:
        raise FileNotFoundError(
            f"No OISST raw files were found under {raw_dir} for JJA {start_year}-{end_year}."
        )
    return files


def load_daily_oisst_on_target(path: Path, target_field: xr.DataArray) -> tuple[pd.Timestamp, xr.DataArray]:
    with xr.open_dataset(path) as ds:
        sst = ds["sst"]
        if "zlev" in sst.dims and sst.sizes["zlev"] == 1:
            sst = sst.squeeze("zlev", drop=True)
        sst = sst.sel(
            lat=slice(PANEL_EXTENT[2], PANEL_EXTENT[3]),
            lon=slice(PANEL_EXTENT[0], PANEL_EXTENT[1]),
        ).interp(
            lat=target_field["lat"],
            lon=target_field["lon"],
            method="linear",
        )
        date = pd.Timestamp(pd.to_datetime(sst["time"].values[0]).normalize())
        return date, sst.isel(time=0, drop=True).astype(np.float32).load()


def load_or_build_oisst_jja_climatology(
    repo_root: Path,
    target_field: xr.DataArray,
) -> xr.DataArray:
    processed_dir = repo_root / "data" / "processed"
    raw_dir = repo_root / "data" / "raw" / "oisst" / "v2.1"
    clim_path = processed_dir / "oisst_jja_1991_2020_jja_mean_on_uwnd_grid.nc"

    if clim_path.exists():
        with xr.open_dataset(clim_path) as ds:
            return ds["sst_climatology"].load().sortby("lat").sortby("lon")

    files = oisst_jja_file_list(raw_dir, SST_CLIM_START_YEAR, SST_CLIM_END_YEAR)
    clim_sum = None
    clim_count = None
    for idx, path in enumerate(files, start=1):
        _, sst = load_daily_oisst_on_target(path, target_field)
        values = sst.values.astype(np.float64)
        valid = np.isfinite(values)
        if clim_sum is None:
            clim_sum = np.zeros_like(values, dtype=np.float64)
            clim_count = np.zeros_like(values, dtype=np.int32)

        clim_sum += np.where(valid, values, 0.0)
        clim_count += valid.astype(np.int32)
        if idx % 300 == 0 or idx == len(files):
            print(f"[OISST climatology] processed {idx}/{len(files)} daily files")

    if clim_sum is None or clim_count is None:
        raise RuntimeError("No OISST files were processed for the requested climatology period.")

    with np.errstate(invalid="ignore", divide="ignore"):
        clim_values = np.where(clim_count > 0, clim_sum / clim_count, np.nan)

    sst_climatology = xr.DataArray(
        clim_values.astype(np.float32),
        coords={"lat": target_field["lat"], "lon": target_field["lon"]},
        dims=("lat", "lon"),
        name="sst_climatology",
    )
    sst_climatology.name = "sst_climatology"
    sst_climatology.attrs.update(
        {
            "long_name": "OISST JJA climatological mean SST",
            "units": "degC",
            "baseline_period": f"{SST_CLIM_START_YEAR}-{SST_CLIM_END_YEAR} JJA",
        }
    )
    sst_climatology.to_dataset(name="sst_climatology").to_netcdf(clim_path)
    return sst_climatology


def load_or_build_olr_jja_anomaly(repo_root: Path) -> tuple[xr.DataArray, xr.DataArray]:
    processed_dir = repo_root / "data" / "processed"
    raw_path = repo_root / "data" / "raw" / "ncar" / "olr.day.mean.nc"
    anom_path = processed_dir / "olr_jja_1991_2023_anom_1991_2020_jja_mean.nc"
    clim_path = processed_dir / "olr_jja_1991_2020_jja_mean.nc"

    if anom_path.exists() and clim_path.exists():
        with xr.open_dataset(clim_path) as ds:
            olr_climatology = ds["olr_climatology"].load().sortby("lat").sortby("lon")
        return open_field(anom_path, "olr_anom"), olr_climatology

    olr_raw = open_field(raw_path, "olr")
    jja = olr_raw.sel(time=olr_raw["time"].dt.year <= OLR_END_YEAR)
    jja = jja.sel(time=jja["time"].dt.month.isin(JJA_MONTHS))
    baseline = jja.sel(time=jja["time"].dt.year <= OLR_CLIM_END_YEAR)

    olr_climatology = baseline.mean("time").astype(np.float32)
    olr_climatology.name = "olr_climatology"
    olr_climatology.attrs.update(
        {
            "long_name": "CPC daily OLR JJA climatological mean",
            "units": "W m-2",
            "baseline_period": f"{OLR_CLIM_START_YEAR}-{OLR_CLIM_END_YEAR} JJA",
        }
    )

    olr_anom = (jja - olr_climatology).astype(np.float32)
    olr_anom.name = "olr_anom"
    olr_anom.attrs.update(
        {
            "long_name": "Daily OLR anomaly relative to 1991-2020 JJA mean",
            "units": "W m-2",
            "baseline_period": f"{OLR_CLIM_START_YEAR}-{OLR_CLIM_END_YEAR} JJA mean",
        }
    )

    olr_anom.to_dataset(name="olr_anom").to_netcdf(anom_path)
    olr_climatology.to_dataset(name="olr_climatology").to_netcdf(clim_path)
    return olr_anom, olr_climatology


def build_oisst_ssta_node_composites(
    repo_root: Path,
    target_field: xr.DataArray,
    assignments: pd.DataFrame,
    sst_climatology: xr.DataArray,
) -> xr.Dataset:
    raw_dir = repo_root / "data" / "raw" / "oisst" / "v2.1"
    files = oisst_jja_file_list(raw_dir, SST_CLIM_START_YEAR, SST_END_YEAR)
    assignment_map = assignments.set_index("date")["node"].to_dict()
    template = np.zeros((target_field.sizes["lat"], target_field.sizes["lon"]), dtype=np.float64)
    count_template = np.zeros((target_field.sizes["lat"], target_field.sizes["lon"]), dtype=np.int32)
    clim_values = sst_climatology.values.astype(np.float64)

    sum_by_node = {node: template.copy() for node in ALL_NODES}
    sumsq_by_node = {node: template.copy() for node in ALL_NODES}
    count_by_node = {node: count_template.copy() for node in ALL_NODES}

    for idx, path in enumerate(files, start=1):
        date, sst = load_daily_oisst_on_target(path, target_field)
        node = assignment_map.get(date)
        if node is None:
            continue

        anomaly = sst.values.astype(np.float64) - clim_values
        valid = np.isfinite(anomaly)
        sum_by_node[node] += np.where(valid, anomaly, 0.0)
        sumsq_by_node[node] += np.where(valid, anomaly * anomaly, 0.0)
        count_by_node[node] += valid.astype(np.int32)

        if idx % 300 == 0 or idx == len(files):
            print(f"[OISST composites] processed {idx}/{len(files)} daily files")

    comp_list = []
    std_list = []
    se_list = []
    n_list = []
    sig2_list = []
    for node in ALL_NODES:
        n = count_by_node[node]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, sum_by_node[node] / n, np.nan)
            variance = np.where(n > 0, sumsq_by_node[node] / n - mean * mean, np.nan)
        variance = np.where(np.isfinite(variance), np.maximum(variance, 0.0), np.nan)
        std = np.sqrt(variance)
        se = np.where(n > 0, std / np.sqrt(n), np.nan)
        sig2 = np.where(n > 0, np.abs(mean) >= 2.0 * se, False)

        comp_list.append(mean.astype(np.float32))
        std_list.append(std.astype(np.float32))
        se_list.append(se.astype(np.float32))
        n_list.append(n.astype(np.int32))
        sig2_list.append(sig2.astype(np.int8))

    coords = {"node": ALL_NODES, "lat": target_field["lat"], "lon": target_field["lon"]}
    return xr.Dataset(
        {
            "sst_anom_comp": xr.DataArray(np.stack(comp_list), coords=coords, dims=("node", "lat", "lon")),
            "sst_anom_std": xr.DataArray(np.stack(std_list), coords=coords, dims=("node", "lat", "lon")),
            "sst_anom_se": xr.DataArray(np.stack(se_list), coords=coords, dims=("node", "lat", "lon")),
            "sst_anom_n": xr.DataArray(np.stack(n_list), coords=coords, dims=("node", "lat", "lon")),
            "sst_anom_sig2": xr.DataArray(np.stack(sig2_list), coords=coords, dims=("node", "lat", "lon")),
            "sst_climatology_1991_2020_jja_mean": sst_climatology,
        }
    )


def plot_shaded_node_panels(
    comp: xr.DataArray,
    sig2: xr.DataArray,
    counts: xr.DataArray,
    out_fig: Path,
    *,
    extent: list[float],
    vmin: float,
    vmax: float,
    cbar_label: str,
    suptitle: str | None = None,
) -> None:
    proj = ccrs.PlateCarree()
    lon_min, lon_max, lat_min, lat_max = extent
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14.0, 7.8),
        layout="constrained",
        subplot_kw={"projection": proj},
    )

    pcm = None
    for node, ax in zip(ALL_NODES, axes.flat):
        # Use the shifted PlateCarree coordinates directly so wrapped extents
        # do not get reinterpreted by cartopy when centering the panel at 100E.
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.coastlines(linewidth=0.8)
        add_lonlat_gridlines(ax, extent, xlocs=PANEL_XTICKS, xlabels=PANEL_XTICK_LABELS, ylocs=PANEL_YTICKS)


        da = comp.sel(node=node).squeeze(drop=True)
        mask = sig2.sel(node=node).squeeze(drop=True)
        n_values = counts.sel(node=node).values
        node_n = int(np.nanmax(n_values)) if np.isfinite(n_values).any() else 0

        if np.isfinite(da.values).any():
            pcm = da.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                add_colorbar=False,
                add_labels=False,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                extend="both",
            )
            ax.contourf(
                da["lon"].values,
                da["lat"].values,
                mask.astype(int).values,
                levels=[0.5, 1.5],
                hatches=["...."],
                colors="none",
                transform=ccrs.PlateCarree(),
            )

        ax.set_title(f"Node {node} (n={node_n})", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if pcm is not None:
        cbar = fig.colorbar(
            pcm,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.05,
            pad=0.03,
            aspect=50,
            extend="both",
        )
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate SOM node/composite panel figures with lon/lat labels.")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd())
    outdir = repo_root / "outputs" / "som_u850"
    outdir.mkdir(parents=True, exist_ok=True)

    assignments = load_assignments(outdir)

    uwnd = subset_panel_domain(
        open_field(repo_root / "data" / "processed" / "uwnd_z850_jja_1991_2023.nc", "uwnd"),
        PANEL_EXTENT,
    )
    uwnd_clim = uwnd.groupby("time.dayofyear").mean("time")
    uwnd_anom = uwnd.groupby("time.dayofyear") - uwnd_clim
    uwnd_sel = align_field_with_nodes(uwnd_anom, assignments)
    uwnd_comp, uwnd_std, uwnd_n, uwnd_sig2 = composite_by_node(uwnd_sel)
    xr.Dataset(
        {
            "uwnd_anom_comp": uwnd_comp,
            "uwnd_anom_std": uwnd_std,
            "uwnd_anom_n": uwnd_n,
            "uwnd_anom_se": uwnd_std / np.sqrt(uwnd_n),
            "uwnd_anom_sig2": uwnd_sig2,
            "uwnd_climatology": uwnd_clim,
        }
    ).to_netcdf(outdir / "u850_anom_node_composites.nc")
    plot_shaded_node_panels(
        uwnd_comp,
        uwnd_sig2,
        uwnd_n,
        outdir / "u850_anom_node_composites_sig2.png",
        extent=PANEL_EXTENT,
        vmin=-4.0,
        vmax=4.0,
        cbar_label="850 hPa zonal wind anomaly (m s-1)",
    )

    olr_anom, olr_climatology = load_or_build_olr_jja_anomaly(repo_root)
    olr = subset_panel_domain(olr_anom, PANEL_EXTENT)
    olr_sel = align_field_with_nodes(olr, assignments)
    olr_comp, olr_std, olr_n, olr_sig2 = composite_by_node(olr_sel)
    xr.Dataset(
        {
            "olr_anom_comp": olr_comp,
            "olr_anom_std": olr_std,
            "olr_anom_n": olr_n,
            "olr_anom_se": olr_std / np.sqrt(olr_n),
            "olr_anom_sig2": olr_sig2,
            "olr_climatology_1991_2020_jja_mean": subset_panel_domain(olr_climatology, PANEL_EXTENT),
        }
    ).to_netcdf(outdir / "olr_anom_node_composites.nc")
    plot_shaded_node_panels(
        olr_comp,
        olr_sig2,
        olr_n,
        outdir / "olr_anom_node_composites_sig2.png",
        extent=PANEL_EXTENT,
        vmin=-20.0,
        vmax=20.0,
        cbar_label="OLR anomaly",
    )

    sst_climatology = load_or_build_oisst_jja_climatology(
        repo_root,
        uwnd,
    )
    sst_ds = build_oisst_ssta_node_composites(repo_root, uwnd, assignments, sst_climatology)
    sst_ds.to_netcdf(outdir / "sst_anom_node_composites.nc")
    plot_shaded_node_panels(
        sst_ds["sst_anom_comp"],
        sst_ds["sst_anom_sig2"],
        sst_ds["sst_anom_n"],
        outdir / "sst_anom_node_composites_sig2.png",
        extent=PANEL_EXTENT,
        vmin=-1.0,
        vmax=1.0,
        cbar_label="SST anomaly relative to 1991-2020 JJA mean (degC)",
    )

    print(outdir / "u850_anom_node_composites_sig2.png")
    print(outdir / "olr_anom_node_composites_sig2.png")
    print(outdir / "sst_anom_node_composites_sig2.png")


if __name__ == "__main__":
    main()
