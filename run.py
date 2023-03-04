# main script to run example usage
import argparse
parser = argparse.ArgumentParser()
# add argumnet --N of type int default 2
parser.add_argument('--N', type=int, default=2)
# parse arguments
args = parser.parse_args()
# constants
N = args.N

import numpy as np
from PIL import Image
spiral = np.asarray(Image.open('samples/spiral.png'))
cat = np.asarray(Image.open('samples/Cat.png'))
spiral_grey = np.asarray(Image.open('samples/spiral.png').convert('L'))
print(spiral_grey.shape)
print(spiral.shape)

from wfc.wfc_tiler import TileDataset
from wfc.wfc_solver import WaveSolver, SimulatedAnnealingSolver 
import matplotlib.pyplot as plt

td_HW = TileDataset(bitcube=spiral_grey.astype(int), form='HW',  N=N, wrap_pad=True) # compare tiles
wavyHW = WaveSolver(td_HW, form='HW', output_size=(50, 50), wrap_pad=True) # initialize wave solver or simulated annealing solver
out_HW = wavyHW.solve() # solve
plt.imshow(out_HW) # plot
plt.savefig('HW_test.png') # save

td_HWC = TileDataset(bitcube=cat.astype(int), form='HWC',  N=N, wrap_pad=True)
td_HWC.cubes = td_HWC.cubes.astype(np.uint16)
wavyHWC = WaveSolver(td_HWC, output_size=(50, 50))
out_HWC = wavyHWC.solve()
plt.imshow(out_HWC)
plt.savefig('HWC_test.png')
