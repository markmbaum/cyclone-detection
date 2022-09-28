"""
This script pulls files from GCS and simply concatenates them from daily files into monthly files. It works on the results of restructure_inputs.py and is the second processing step.
"""
# %%

import os
import pandas as pd
import numpy as np
import multiprocessing

# %%

def combine_inputs(year, month):

    #download files
    os.system(f'gsutil -qm cp gs://cyclone-tracking/{year}_{month}_*.npy .')
    #number of days in target month
    days = pd.Period(f'{year}-{month}').days_in_month
    #load files into memory, in order
    X = [np.load(f'{year}_{month}_{day}.npy') for day in range(1,days+1)]
    #stack 'em all up along the temporal dimension
    X = np.concatenate(X, axis=0)
    #write the big array to file
    np.save(f'{year}_{month}.npy', X)
    #send it back to GCS
    os.system(f'gsutil -qm cp {year}_{month}.npy gs://cyclone-tracking/{year}_{month}.npy')
    #remove the daily files and the new file
    os.system(f'rm {year}_{month}*.npy')
    
    print(f'{year}-{month} complete')

    return None

# %%

if __name__ == '__main__':

    pool = multiprocessing.Pool()
    tasks = []
    for year in range(1980, 2022):
        for month in range(1, 13):
            tasks.append(
                pool.apply_async(
                    combine_inputs,
                    (year, month)
                )
            )
    [task.get() for task in tasks]
    pool.close()