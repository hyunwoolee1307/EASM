from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from easm_cwes_utils import (
    find_repo_root,
    open_field,
    align_on_common_time,
    make_mmdd_anomaly,
    subset_domain,
)


def _flatten_domain(da):
    domain = subset_domain(da).sortby("lat", ascending=False).sortby("lon")
    flat = domain.stack(cell=("lat", "lon")).transpose("time", "cell")
    return domain, flat


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    processed_dir = repo_root / "data" / "processed"
    outdir = repo_root / "outputs" / "som_u850_olr" / "preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)

    uwnd = open_field(processed_dir / "uwnd_z850_jja_1991_2023.nc", "uwnd")
    olr = open_field(processed_dir / "olr_jja_1991_2023.nc", "olr")
    uwnd, olr = align_on_common_time(uwnd, olr)

    uwnd_anom, _ = make_mmdd_anomaly(uwnd)
    conv = -olr
    conv.name = "conv"

    u_domain, u_flat = _flatten_domain(uwnd_anom)
    c_domain, c_flat = _flatten_domain(conv)

    if not np.array_equal(u_domain["time"].values, c_domain["time"].values):
        raise RuntimeError("u850 and convection fields are not aligned on time.")

    dates = pd.to_datetime(u_domain["time"].values).normalize()
    lat_grid, lon_grid = np.meshgrid(
        u_domain["lat"].values,
        u_domain["lon"].values,
        indexing="ij",
    )
    weight = np.sqrt(np.cos(np.deg2rad(lat_grid)))

    dates_df = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
        }
    )
    dates_df.to_csv(outdir / "som_dates.csv", index=False)

    cell_meta = pd.DataFrame(
        {
            "cell_id": np.arange(1, u_flat.sizes["cell"] + 1),
            "lat": lat_grid.reshape(-1),
            "lon": lon_grid.reshape(-1),
            "weight": weight.reshape(-1),
        }
    )
    cell_meta.to_csv(outdir / "som_cell_metadata.csv", index=False)

    u_df = pd.DataFrame(u_flat.values, columns=[f"cell_{i}" for i in cell_meta["cell_id"]])
    c_df = pd.DataFrame(c_flat.values, columns=[f"cell_{i}" for i in cell_meta["cell_id"]])

    u_df.to_csv(outdir / "u850_anom_domain.csv", index=False)
    c_df.to_csv(outdir / "conv_domain.csv", index=False)

    print(f"Exported {len(dates_df)} daily samples and {len(cell_meta)} domain cells to {outdir}")


if __name__ == "__main__":
    main()
