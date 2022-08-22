# %%

import zarr
import numpy as np
import dask.array as da

# %%

z = zarr.open(
    '../data/test.zarr',
    mode='w',
    shape=(0,720,1440,3),
    dtype=np.float32,
    fill_value=np.float32(np.nan)
)
# %%

for month in range(8,11):
    #load month of 6 hourly vorticity, pressure, and temperature
    x = np.load(f'../data/pro/inputs/2010_{month}_inputs.npy')
    #swap channel dimension to last dimension
    x = np.moveaxis(x, 1, -1)
    #get rid of a pole point for even numbers in both dimensions
    x = x[:,:-1,...]
    #standardize
    m = x.astype(np.float64).mean(axis=0).astype(np.float32)
    s = x.astype(np.float64).std(axis=0).astype(np.float32)
    for i in range(x.shape[0]):
        x[i,...] = (x[i,...] - m)/s
    print(x.mean())
    print(x.std())
    #append to the zarray
    z.append(x)

# %%

z = da.from_zarr('../data/test.zarr')
# %%
