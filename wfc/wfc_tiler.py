# space to implement 3D tiler class
import numpy as np
import sys
from tqdm import tqdm
import os
import wfc.utils_tiler as ut


class TileDataset:

    def __init__(self, form='HWD', bitcube=None, N=None, wrap_pad=True): 
        self.bitcube = bitcube
        self.N = N
        self.O = N-1
        self.form = form
        self.wrap_pad = wrap_pad
        
        # extract all overlapping cubes
        bitcube_pad, cube_master = ut.extract_cubes(bitcube, form, N=N, wrap_pad=wrap_pad)
        cube_master = cube_master
        print(f'cube_master dtype {cube_master.dtype}')
        
        # if wrap_pad, remove padding of np nans

        # get all unique cubes
        self.cubes, self.index, self.counts = np.unique(cube_master ,return_counts=True ,return_index=True, axis=0)
        print(f'unique cubes: {len(self.cubes)}')

        # calculate constraints
        self.constraints = ut.compare_cubes(self.cubes, form, N=N)

    def __len__(self):
        return len(self.cubes)
