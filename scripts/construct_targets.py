# %%

from os.path import join
import json
import pandas as pd
import numpy as np
import multiprocessing

# %%

DIROUT = join('..', 'data', 'pro', 'targets')

FNTRACKS = join('..', 'data', 'raw', 'ibtracs.ALL.list.v04r00.csv')

YEARS = [2010, 2020]

MONTHS = range(8, 11)

# %%

def wrap_longitude(lon):
    assert -180 <= lon <= 360, f"longitude {lon} out of expected bounds"
    if -180 <= lon < 0:
        return lon + 360
    return lon

def bump(lat, lon, Lat, Lon, r=5):
    d = np.sqrt((lat - Lat)**2 + (lon - Lon)**2)
    f = np.exp(-d/(2*r))/2
    return f

def construct_month_inputs(year, month, tracks):

    #timestamped boundaries of the month
    tstart = pd.Timestamp(f'{year}-{month}-01')
    tend = pd.Timestamp(f'{year}-{month+1}-01')
    #construct a date range for 6 hourly slices through the month
    time = pd.date_range(tstart, tend, inclusive='left', freq='6H')
    #slice out the month's hurricane tracks
    tracks = tracks[(tracks.time >= tstart) & (tracks.time < tend)].copy()
    #coordinate grids
    lat = np.linspace(90, -90, 721, dtype=np.float32)
    lon = np.linspace(0, 360, 1440, endpoint=False, dtype=np.float32)
    Lon, Lat = np.meshgrid(lon, lat)
    #stack of target arrays/images
    target_maps = np.zeros((len(time), 721, 1440), dtype=np.float32)
    #stack of binary classification flags (yes cyclones/no cyclones)
    target_flags = np.zeros(len(time), dtype=np.bool_)
    #handle longitude convention
    tracks['lon'] = tracks['lon'].map(wrap_longitude)
    #a list of metadata
    meta = [[str(t)] for t in time]

    for i,t in enumerate(time):
        sl = tracks[tracks.time == t]
        sl.index = range(len(sl))
        if not sl.empty:
            target_flags[i] = True
            for idx in sl.index:
                lat, lon = sl[['lat','lon']].loc[idx].values
                target_maps[i,...] += bump(lat, lon, Lat, Lon)
                meta[i].append(
                    {
                        'lat': round(lat, 4),
                        'lon': round(lon, 4),
                        'nature': sl.at[idx,'nature'],
                        'status': sl.at[idx,'status'],
                    }
                )

    #take a polar point off the target maps for even dimension sizes
    target_maps = target_maps[:,:-1,:]
    #expand the dimensions by one axis for easy tensor conversion
    target_maps = np.expand_dims(target_maps, -1)
    print(target_maps.shape)

    return target_maps, target_flags, meta, tracks

def construct_and_write(year, month, tracks):

    maps, flags, meta, tracks = construct_month_inputs(year, month, tracks)

    p = join(DIROUT, f'{year}_{month}_target_maps.npy')
    np.save(p, maps, allow_pickle=False)

    p = join(DIROUT, f'{year}_{month}_target_flags.npy')
    np.save(p, flags, allow_pickle=False)

    p = join(DIROUT, f'{year}_{month}_meta.json')
    with open(p, 'w') as ofile:
        json.dump(meta, ofile, indent=True)

    return None

# %%

if __name__ == '__main__':

    tracks = pd.read_csv(FNTRACKS, skiprows=[1], low_memory=False)

    #provisional spurs get removed 
    tracks = tracks[tracks['TRACK_TYPE'] != 'PROVISIONAL_spur']

    #select and rename columns
    tracks = tracks[['LAT', 'LON', 'ISO_TIME', 'NATURE', 'USA_STATUS']]
    tracks.columns = ['lat', 'lon', 'time', 'nature', 'status']

    #take only certain storms
    tracks = tracks[tracks.nature == 'TS']
    #tracks = tracks[np.isin(tracks.status, ('HU', 'HR', 'TC'))]

    #convert datatypes
    tracks.time = tracks.time.map(pd.Timestamp)
    for c in ('lat', 'lon'):
        #can't do anything without both coordinates
        tracks = tracks[tracks[c] != ' ']
        tracks[c] = tracks[c].map(np.float32)

    #clean the status strings
    tracks.nature = tracks['nature'].map(lambda s: s.strip())

    #reset integer index
    tracks.index = range(len(tracks))

    #generate target maps and metadata in parallel
    pool = multiprocessing.Pool()
    tasks = []
    for year in YEARS:
        for month in MONTHS:
            tasks.append(
                pool.apply_async(
                    construct_and_write,
                    (year, month, tracks)
                )
            )
    [task.get() for task in tasks]

# %%

# %%
