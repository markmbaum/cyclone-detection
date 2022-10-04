from os.path import join
import numpy as np

__all__ = [
    'split_flip_stack'
]

#flips and stacks the southern hemisphere onto the northern for a whole month (3D array)
def split_flip_stack(x, negate_southern_channels=[]):
    north, south = np.split(x, 2, axis=1)
    south = np.flip(south, axis=1)
    for channel in negate_southern_channels:
        south[:,:,:,channel] *= -1
    x = np.concatenate([north, south], axis=0)
    return x
