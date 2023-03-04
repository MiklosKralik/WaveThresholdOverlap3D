# Python Impelemtation of Wave Function Collapse
The goal of this repository is to create an innovative collection of solvers for the [WFC](https://github.com/mxgmn/WaveFunctionCollapse)
algorithm. These could be energy-based or CSP(constraints solving problem)-based solvers. Contributions welcome!

### Future Plans
-- Speed up the current (very slow) pairwise comparison of tiles
-- Update simulated annealing solver pertubation strategy
-- Implement GPU slovers with Cupy or pytorch
-- And, of course, add many more interesting solvers!!

### Core Objects:
- TileDataset: Class for storing and transforming tiles extracted from 2D or 3D inputs
- WaveSolver, SimulatedAnnealingSolver: Classes for implementing brute force and annealing solver

### Example Usage:
```
from wfc.wfc_tiler import TileDataset
from wfc.wfc_solver import WaveSolver, SimulatedAnnealingSolver 
import matplotlib.pyplot as plt

td_HW = TileDataset(bitcube=spiral_grey.astype(int), form='HW',  N=N, wrap_pad=True) # compare tiles
wavyHW = WaveSolver(td_HW, form='HW', output_size=(50, 50), wrap_pad=True) # initialize wave solver or simulated annealing solver
out_HW = wavyHW.solve() # solve
plt.imshow(out_HW) # plot
plt.savefig('HW_test.png') # save
```
### Requirements
python >= 3.9
numpy >= 1.21
matplotlib >= 3.5
tqdm >= 4.64
scipy >= 1.10
pillow >= 9.0

#### Miklos Kralik miklos.a.kralik@gmail.com


