# %%

from os.path import join
import json
import numpy as np
import matplotlib.pyplot as plt

# %%

lat = np.linspace(90, -90, 721, dtype=np.float32)
lon = np.linspace(0, 360, 1440, endpoint=False, dtype=np.float32)

# %%

#load a month of data
datadir = join('..', 'data', 'pro')
year, month = 2010, 9
inputs = np.load(join(datadir, 'inputs', f'{year}_{month}_inputs.npy'))
targets = np.load(join(datadir, 'targets', f'{year}_{month}_targets.npy'))
with open(join(datadir, 'targets', f'{year}_{month}_coords.json'), 'r') as ifile: 
    coords = json.load(ifile)

# %%

idx = [i for i in range(len(coords)) if len(coords[i]) > 1]

# %%

i = idx[1]

fig, axs = plt.subplots(2, 1, figsize=(10,10), constrained_layout=True)

channel = 0
axs[0].pcolormesh(lon, lat, inputs[i,channel,...] - inputs[:,channel,...].mean(axis=0))
axs[1].pcolormesh(lon, lat, targets[i,...])

for d in coords[i][1:]:
    for ax in axs:
        ax.plot(
            d['lon'],
            d['lat'],
            'ro',
            markerfacecolor='none',
            markersize=30,
            alpha=0.25
        )

axs[0].set_title(f'{coords[i][0]} Input Channel {channel}')
axs[1].set_title('Target Image')
fig.supxlabel('Longitude')
fig.supylabel('Latitude')

#fig.savefig('../plots/input_target_sample', dpi=400)
# %%
