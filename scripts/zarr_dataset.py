import os
from os.path import join
import zarr
import numpy as np

from utils import *

YEARS = range(2000, 2022)

MONTHS = range(1, 13)

BUCKET = 'gs://cyclone-tracking/'

fninputs = lambda year, month: join('..', 'data', 'pro', 'inputs', f'{year}_{month}_inputs.npy')

fntargets = lambda year, month: join('..', 'data', 'pro', 'targets', f'{year}_{month}_target_maps.npz')

readinputs = lambda year, month: np.load(fninputs(year, month))

readtargets = lambda year, month: np.load(fntargets(year, month))['maps']

def generate_classifications(Y, dx=2**5, dy=2**5):
    N = 256//dy
    M = 1440//dx
    C = np.zeros((Y.shape[0], N, M, 1), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            i1, i2 = i*dy, (i+1)*dy
            j1, j2 = j*dx, (j+1)*dx
            C[:,i,j,0] = Y[:,i1:i2,j1:j2,0].sum(axis=(1,2))
    return C

for year in YEARS:

    #determine 0th axis size
    L = sum([np.load(fninputs(year, month), mmap_mode='r').shape[0] for month in MONTHS])
    print(f'{L} time slices in {year}, {2*L} after splitting')

    fnX = f'{year}_inputs.zarr'
    Xout = zarr.open(
        fnX,
        mode='w',
        shape=(2*L,256,1440,3),
        chunks=(1,256,1440,3),
        compressor=None,
        dtype=np.float32
    )

    fnY = f'{year}_targets.zarr'
    Yout = zarr.open(
        fnY,
        mode='w',
        shape=(2*L,8,45,1),
        compressor=None,
        dtype=np.float32
    )

    N = 0
    for month in MONTHS:
       
        print(f'starting {year}-{month}')

        X = readinputs(year, month)
        X = np.moveaxis(X, 1, -1)
        X = X[:,104:-105,...]
        X = (X - X.mean(axis=0))/X.std(axis=0)
        X = split_flip_stack(X, [0])
        for i in range(X.shape[0]):
            x = X[i,...]
            X[i,...] = (x - x.mean())/x.std()

        Y = readtargets(year, month)
        Y = Y[:,104:-105,...]
        Y = split_flip_stack(Y)
        Y = generate_classifications(Y)
        
        n = X.shape[0]
        Xout[N:N+n,...] = X
        Yout[N:N+n,...] = Y
        print(f'{year}-{month} slices written into {N}:{N+n} on axis 0')
        N += n
   
    for fn in (fnX, fnY):
        print(f'copying {fn} to GCS')
        os.system(f'gsutil -qm cp -r {fn} {BUCKET}')
        print(f'deleting local {fn}')
        os.system(f'rm -r {fn}')