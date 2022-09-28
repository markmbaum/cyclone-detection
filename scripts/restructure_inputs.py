"""
This script pulls the original ERA5 reanalysis files from the appropriate GCS bucket, slices them at 6-hourly intervals, stacks them into a single array, the sends them back to GCS. Its the first processing step for the reanalysis files.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing

def restructure_inputs(year, month, day):
    
    #download files
    os.system(f'gsutil -q cp gs://era-5/{year}_{month}_{day}.nc .')
    os.system(f'gsutil -q cp gs://era-5/{year}_{month}_{day}_pressure.nc .')
    #load files
    fields = xr.load_dataset(f'{year}_{month}_{day}.nc')
    p = xr.load_dataset(f'{year}_{month}_{day}_pressure.nc')
    #concatenate into a single array
    X = np.stack([
        fields.vo.values[::6,1,...], #vorticity at 850 mbar
        fields.t.values[::6,0,...], #temperature at 500 mbar
        p.sp.values[::6,...] #surface pressure
    ], axis=1)
    #write the array to file
    np.save(f'{year}_{month}_{day}.npy', X, allow_pickle=False)
    #write the array back to GCS
    os.system(f'gsutil -q cp {year}_{month}_{day}.npy gs://cyclone-tracking-inputs/{year}_{month}_{day}.npy')
    #delete the files
    os.system(f'rm {year}_{month}_{day}.nc')
    os.system(f'rm {year}_{month}_{day}_pressure.nc')
    os.system(f'rm {year}_{month}_{day}.npy')

    return None

if __name__ == '__main__':

    pool = multiprocessing.Pool()
    tasks = []
    for year in range(1980, 2022):
        for month in range(1, 13):
            for day in range(1, pd.Period(f'{year}-{month}').days_in_month + 1):
                tasks.append(
                    pool.apply_async(
                        restructure_inputs,
                        (year, month, day)
                    )
                )
    [task.get() for task in tasks]
    pool.close()