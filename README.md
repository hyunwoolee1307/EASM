# EASM SOM Workflow

This repository contains a notebook-first workflow for preprocessing atmospheric and oceanic datasets, running self-organizing map (SOM) analyses on JJA 850 hPa zonal wind and joint `u850 + OLR` anomalies, and generating follow-up diagnostics for wind, convection, SST, and EASM-CWES relationships.

## Repository Layout

```text
EASM/
├── README.md
├── data/
│   ├── raw/
│   │   ├── ncar/
│   │   └── oisst/
│   │       └── v2.1/
│   └── processed/
├── notebooks/
│   ├── preprocessing/
│   │   ├── concat_uwnd.ipynb
│   │   ├── concat_vwnd.ipynb
│   │   ├── concat_sst.ipynb
│   │   ├── crop_olr.ipynb
│   │   └── regrid_oisst.ipynb
│   └── analysis/
│       ├── plot_uwnd.ipynb
│       ├── plot_oisst.ipynb
│       ├── composite_olr.ipynb
│       ├── plot_u850_olr_som.ipynb
│       ├── validate_easm_cwes.ipynb
│       └── interdecadal_wnp_nio.ipynb
├── outputs/
│   ├── som_u850/
│   └── som_u850_olr/
└── scripts/
    ├── easm_cwes_utils.py
    ├── export_som_u850_olr_inputs.py
    ├── run_kohonen.R
    └── run_som_u850_olr.R
```

## Data Contract

Raw inputs are expected in the following locations:

- `data/raw/ncar/uwnd.????.nc`
- `data/raw/ncar/vwnd.????.nc`
- `data/raw/ncar/olr.day.anom.nc`
- `data/raw/oisst/v2.1/YYYYMM/*.nc`

Processed files are written to `data/processed/`:

- `uwnd_z850_jja_1991_2023.nc`
- `vwnd_z850_jja_1991_2023.nc`
- `oisst_jja_1991_2023.nc`
- `oisst_jja_1991_2023_on_uwnd_grid.nc`
- `olr_jja_1991_2023.nc`

SOM outputs are written to `outputs/som_u850/` and `outputs/som_u850_olr/`, including:

- `som_daily_assignment.csv`
- SOM diagnostics and composite figures
- SOM model object (`som_model.rds`)
- Derived NetCDF composite products
- Annual occurrence frequencies and EASM-CWES validation tables

Large raw datasets and generated outputs are intentionally untracked and ignored by `.gitignore`.

## Environment Setup

### Conda / Python

The expected execution environment for the analysis notebooks is the existing `ocpc` conda environment:

```bash
conda run -n ocpc python -c "import xarray, pandas, numpy, matplotlib, cartopy, xesmf"
```

When running plotting code headlessly, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

If you are rebuilding the environment elsewhere, install Python packages equivalent to `xarray`, `pandas`, `numpy`, `matplotlib`, `cartopy`, and `xesmf`.

### R

Install the required R packages:

```r
install.packages(c("terra", "kohonen", "ncdf4"))
```

## Workflow

Run the workflow in this order:

1. Execute the preprocessing notebooks in `notebooks/preprocessing/`.
2. Run the original `u850` SOM script if you need the baseline branch:

```bash
Rscript scripts/run_kohonen.R
```

3. Run the multivariate `u850 + OLR` SOM branch:

```bash
conda run -n ocpc Rscript scripts/run_som_u850_olr.R
```

4. Execute the analysis notebooks in `notebooks/analysis/`.

For the new multivariate branch, run:

```bash
conda run -n ocpc jupyter nbconvert --to notebook --execute notebooks/analysis/plot_u850_olr_som.ipynb
conda run -n ocpc jupyter nbconvert --to notebook --execute notebooks/analysis/validate_easm_cwes.ipynb
conda run -n ocpc jupyter nbconvert --to notebook --execute notebooks/analysis/interdecadal_wnp_nio.ipynb
```

All notebooks and the R script include lightweight repository-root bootstrapping, so they can be launched from the repository root or from their own subdirectories.

## Notebook Roles

- `concat_uwnd.ipynb`: combine NCAR `uwnd` files and export JJA 850 hPa data.
- `concat_vwnd.ipynb`: combine NCAR `vwnd` files and export JJA 850 hPa data.
- `concat_sst.ipynb`: combine daily OISST anomaly files and export JJA-only data.
- `crop_olr.ipynb`: extract JJA OLR anomaly data for 1991-2023.
- `regrid_oisst.ipynb`: regrid OISST anomalies onto the `uwnd` grid.
- `plot_uwnd.ipynb`: build node-wise 850 hPa zonal wind anomaly composites and save figure/netCDF outputs.
- `plot_oisst.ipynb`: compute SST composites by SOM node.
- `composite_olr.ipynb`: compute OLR composites by SOM node.
- `plot_u850_olr_som.ipynb`: build node diagnostics for the joint `u850 + OLR` SOM, including SST composites and node summaries.
- `validate_easm_cwes.ipynb`: compute `WNPMI`, `IMI`, WNP convection, `CWES`, northern Indian Ocean SST, Indian Ocean basinwide SST, and ENSO-controlled validation products.
- `interdecadal_wnp_nio.ipynb`: summarize inter-decadal changes in node occurrence and their relationships with northern Indian Ocean and WNP local SST variability.

## Notes

- `scripts/easm_cwes_utils.py` centralizes the Python-side anomaly, index, partial-correlation, and FDR logic used by the new notebooks.
- `run_som_u850_olr.R` now reads the processed NetCDF inputs directly with `terra` and `ncdf4`, then writes both NetCDF and flattened CSV versions of the SOM codebook/composite products.
- `scripts/export_som_u850_olr_inputs.py` is kept as a legacy fallback utility but is no longer required in the normal `ocpc` workflow.
- Existing analysis products under `outputs/som_u850/` and `outputs/som_u850_olr/` are treated as working artifacts, not source files.
