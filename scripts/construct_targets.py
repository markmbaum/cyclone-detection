# %%

from os.path import join
import json
import pandas as pd
import numpy as np
import multiprocessing

# %%

DIROUT = join('..', 'data', 'pro', 'targets')

FNTRACKS = join('..', 'data', 'raw', 'ibtracs.ALL.list.v04r00.csv')

# %%

def wrap_longitude(lon):
    assert -180 <= lon <= 180, "longitude out of expected bounds"
    if -180 <= lon < 0:
        return lon + 360
    return lon

def bump(lat, lon, Lat, Lon, r=2):
    d = np.sqrt((lat - Lat)**2 + (lon - Lon)**2)
    f = np.exp(-d/(2*r))
    return f

def construct_month_inputs(year, month, tracks):

    #timestamped boundaries of the month
    tstart = pd.Timestamp(f'{year}-{month}-01')
    tend = pd.Timestamp(f'{year}-{month+1}-01')
    #construct a date range for 6 hourly slices through the month
    time = pd.date_range(tstart, tend, inclusive='left', freq='6H')
    #slice out the month's hurricane tracks
    tracks = tracks[(tracks.time >= tstart) & (tracks.time < tend) & (tracks.status == 'HU')].copy()
    #coordinate grids
    lat = np.linspace(90, -90, 721, dtype=np.float32)
    lon = np.linspace(0, 360, 1440, endpoint=False, dtype=np.float32)
    Lon, Lat = np.meshgrid(lon, lat)
    #stack of target arrays/images
    targets = np.zeros((len(time), 721, 1440), dtype=np.float32)
    #handle longitude convention
    tracks['lon'] = tracks['lon'].map(wrap_longitude)

    coords = [[str(t)] for t in time]
    for i,t in enumerate(time):
        sl = tracks[tracks.time == t]
        sl.index = range(len(sl))
        if not sl.empty:
            for idx in sl.index:
                lat, lon = sl[['lat','lon']].loc[idx].values
                targets[i,...] += bump(lat, lon, Lat, Lon)
                coords[i].append((round(lat, 4), round(lon, 4)))

    return targets, coords, tracks

def construct_and_write(year, month, tracks):

    targets, coords, tracks = construct_month_inputs(year, month, tracks)
    p = join(DIROUT, f'{year}_{month}_targets.npy')
    np.save(p, targets, allow_pickle=False)
    p = join(DIROUT, f'{year}_{month}_coords.json')
    with open(p, 'w') as ofile:
        json.dump(coords, ofile, indent=True)

    return None

# %%

if __name__ == '__main__':

    tracks = pd.read_csv(FNTRACKS, skiprows=[1])
    #select and rename columns
    tracks = tracks[['USA_LAT', 'USA_LON', 'ISO_TIME', 'USA_STATUS']]
    tracks.columns = ['lat', 'lon', 'time', 'status']
    #convert datatypes
    tracks.time = tracks.time.map(pd.Timestamp)
    for c in ('lat', 'lon'):
        #can't do anything without both coordinates
        tracks = tracks[tracks[c] != ' ']
        tracks[c] = tracks[c].map(np.float32)
    #clean the status strings
    tracks.status = tracks.status.map(lambda s: s.strip())
    #reset integer index
    tracks.index = range(len(tracks))

    pool = multiprocessing.Pool()
    tasks = []
    for year in [2010]:
        for month in range(8, 11):
            tasks.append(
                pool.apply_async(
                    construct_and_write,
                    (year, month, tracks)
                )
            )
    [task.get() for task in tasks]

# %%
