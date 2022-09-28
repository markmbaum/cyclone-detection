import os
from os.path import join
import zarr
import numpy as np

YEARS = range(2000, 2022)

MONTHS = range(1, 13)

BUCKET = 'gs://cyclone-tracking/'

fninputs = lambda year, month: join('..', 'data', 'pro', 'inputs', f'{year}_{month}_inputs.npy')

#standardizes a whole month of data 3D array
def standardize_month(X):
    #first standardize along the month's time axis, removing meridional tendencies
    Y = (X - X.mean(axis=0))/X.std(axis=0)
    #then standardize each slice/time individually
    for i in range(Y.shape[0]):
        y = Y[i,...]
        Y[i,...] = (y - y.mean())/y.std()
    return Y

#flips and stacks the southern hemisphere onto the northern for a whole month (3D array)
def split_flip_stack(X):
    north, south = np.split(X, 2, axis=1)
    south = np.flip(south, axis=1)
    Y = np.concatenate([north, south], axis=0)
    return Y


for year in YEARS:

    #determine 0th axis size
    L = sum([np.load(fninputs(year, month), mmap_mode='r').shape[0] for month in MONTHS])
    print(f'{L} time slices in {year}, {2*L} after splitting')

    fn = f'{year}_inputs.zarr'
    Z = zarr.open(
        fn,
        mode='w',
        shape=(2*L,256,1440,3),
        dtype=np.float32,
        fill_value=np.nan
    )

    n = 0
    for month in MONTHS:
       
        print(f'loading {year}-{month}')
        X = np.load(fninputs(year, month))
        X = np.moveaxis(X, 1, -1)
        X = X[:,104:-105,...]
        X = standardize_month(X)
        X = split_flip_stack(X)
        
        Z[n:n+X.shape[0],...] = X
        print(f'{year}-{month} slices written into {n}:{n+X.shape[0]} on axis 0')
        
        n += X.shape[0]
   
    print(f'copying {fn} to GCS')
    os.system(f'gsutil -qm cp -r {fn} {BUCKET}')
    print(f'deleting local {fn}')
    os.system(f'rm -r {fn}')