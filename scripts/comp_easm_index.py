from pathlib import Path
import numpy as np
import xarray as xr

data_dir = Path.home() / "EASM" / "outputs" / "som_u850"
# %% load data
ds = xr.open_dataset(data_dir / "u850_anom_node_composites.nc")

east_asia_tropics = ds["uwnd_anom_comp"].sel(lat=slice(5, 15), lon=slice(90, 130)).mean(dim=["lat", "lon"])
east_asia_subtropics = ds["uwnd_anom_comp"].sel(lat=slice(22.5, 32.5), lon=slice(110, 140)).mean(dim=["lat", "lon"])

index = east_asia_tropics - east_asia_subtropics

print(index.values)

