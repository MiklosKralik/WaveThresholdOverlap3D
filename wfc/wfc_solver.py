# imports
import numpy as np
import sys
from wfc.wfc_tiler import TileDataset
from scipy.stats import entropy
import time
import threading
from tqdm import tqdm

class WaveSolver:
    def __init__(self, td:TileDataset, output_size:tuple):
        
        # extract information about the tilesize output shape & heuristics
        self.td = td # containes all information about the tiles, constraints, and form
        self.output_size = output_size # will be of length 2 if format is HW or HWC
        self.n_tiles = len(td)
        self.mode = 'raise' # constraints will wrap around output else raise
        self.max_contra = 100 # maximum number of allowed backtracks

        # possible directions & form specific attributes
        if self.td.form == 'HW' or self.td.form == 'HWC':
            self.directions = {'left': (-1, 0), 'right': (1, 0), 'up': (0, -1), 'down': (0, 1)} # thinking of a numpy array starting in the upper left corner
        elif self.td.form == 'HWD':
            self.directions = {'left': (-1, 0, 0), 'right': (1, 0, 0), 'up': (0, -1, 0), 'down': (0, 1, 0), 'front': (0, 0, -1), 'back': (0, 0, 1)}

        # create datastructures to store the wave & entropy & neighbor stack
        self.W = np.full(self.output_size + (self.n_tiles,), True, dtype=bool)
        self.H = np.zeros(self.output_size, dtype=float)
        self.S = []

        # metrics to keep track of the collapse
        self.status = 'INIT' # can be INIT RUN FAIL COLLAPSE
        self.prop = 0
        self.iteration = 0
        self.contra = 0
        self.collapsed = 0

    def _update_entropy(self):
        # calculate entropy for the output
        self.H = entropy(self.W, axis=-1)

    def _random_cell_choice(self):
        # choose a random cell that has not been collapsed with the lowest entropy
        # if there are multiple cells with the same entropy, choose one randomly
        min_idxs = np.argwhere(self.H == self.H[np.nonzero(self.H)].min())

        # returns a list of indices
        return tuple(min_idxs[np.random.randint(len(min_idxs))]) # indicies should always be in tuple format

    def _random_cube_collapse(self, coords):
        # picks a random cube idx based on frequency probabilities and collapses it
        available_idxs = np.argwhere(self.W[coords])[:, 0] # take along second dim
        probs = self.td.counts[available_idxs]
        choice = np.random.choice(available_idxs, p=probs / probs.sum())
        self.W[coords] = False
        self.W[coords][choice] = True
    
    def _check_cell_for_collapse(self, cubes_bool):
        if np.count_nonzero(cubes_bool) == 1:
            return True
        else:
            return False

    def _check_for_collapse(self):
        # check if all cells have collapsed, do not check if self.status is FAIL
        if self.status == 'FAIL':
            return 'FAIL'
        collapsed = np.count_nonzero(self.W == True, axis=-1) == 1
        self.collapsed = collapsed.sum()
        if np.all(collapsed):
            return 'COLLAPSE'
        else:
            return 'RUN'

    def _apply_neighbors(self, coords, trans):
        # apply neighbor transformation
        return tuple([sum(x) for x in zip(coords, trans)])

    def _find_neighbors(self, coords):
        # find neighbors of a cell from and return a list of valid ones
        neighbors = {}
        for d, v in self.directions.items():
            ind = self._apply_neighbors(coords, v)
            try:
                flat_index = np.ravel_multi_index(ind, self.output_size, mode=self.mode)
                neighbors[d] = np.unravel_index(flat_index, self.output_size)
            except ValueError:
                neighbors[d] = None
        return neighbors

    def _observe(self):
        # observe wave matrix, collapse cell with the lowest entropy and return coords
        self._update_entropy()
        coords = self._random_cell_choice()
        self._random_cube_collapse(coords)
        return coords


    def _log(self):
        log_count = 1
        while self.status not in ['FAIL', 'COLLAPSE']:
            last_prop = self.prop
            time.sleep(1)
            log_count += 1
            log_prop = f'PROPAGATION: {self.prop:8}'
            log_iter = f'ITER: {self.iteration:8}'
            log_status = f'STATUS: {self.status:8}'
            log_S = f'STACK: {len(self.S):8}'
            log_collapsed = f'COLL: {self.collapsed:8}'
            log_contra = f'CONTRA: {self.contra:8}'
            sys.stdout.write('\r')
            sys.stdout.write(f'{log_prop} {log_iter} {log_status} {log_S} {log_collapsed} {log_contra}')
            sys.stdout.flush()

    def _propagate(self):
        self.prop = 0
        while len(self.S) > 0:
            coords = self.S.pop(-1)
            neighbors = self._find_neighbors(coords)
            source_cubes = self.W[coords]
            for direction, n_coords in neighbors.items():
                # check if direction is available
                if n_coords == None:
                    continue
                destination_cubes = self.W[n_coords]
                # if not collapsed and if not update == contradiction and if not update = destination
                # add to stack
                if self._check_cell_for_collapse(destination_cubes):
                    continue
                adj_cubes = self.td.constraints[direction][source_cubes].any(axis=0)
                n_update = np.bitwise_and(destination_cubes, adj_cubes)
                if np.all(n_update == False):
                    self.status = 'FAIL'
                    return 'FAIL'
                if (n_update == destination_cubes).all():
                    continue

                self.W[n_coords] = n_update
                self.S.append(n_coords)
                self.prop += 1

    def _backtrack(self, state):
        if state == 'record':            
            self.last_W = self.W
            self.last_H = self.H
        elif state == 'reset':
            self.W = self.last_W
            self.S = [] # stack must be emptied
            self.H = self.last_H

    def _collapse_output(self):
        out = np.zeros(self.output_size, dtype=int)
        if self.td.form == 'HWC':
            out = np.zeros(self.output_size + (4,), dtype=int) #change later this hardcodes four channels
        
        collapsed_W = self.W.argmax(axis=-1)
        print('Collapsing output')
        
        if self.td.form == 'HW' or self.td.form == 'HWC':
            for index in np.ndindex(self.output_size):
                out[index] = self.td.cubes[collapsed_W[index], 0, 0]
        
        elif self.td.form == 'HWD':
            for index in np.ndindex(self.output_size):
                out[index] = self.td.cubes[collapsed_W[index], 0, 0, 0]

        self.out = out
        return out

    def solve(self):
        logger = threading.Thread(target=self._log)
        logger.start()
        print('solving...')
        self.status = self._check_for_collapse()
        self.iteration = 0
        self._update_entropy() # first entropy update to populate H
        coords = self._observe() # cell to start prop from
        while self.status not in ['FAIL', 'COLLAPSE']:
            self._backtrack('record')
            self.S.append(coords)
            self._propagate()
            if self.status == 'FAIL' and self.contra < self.max_contra:
                self._backtrack('reset')
                self.contra += 1
                self.status = 'RUN'
            self.status = self._check_for_collapse()
            if self.status not in ['FAIL', 'COLLAPSE']:
                coords = self._observe()
            self.iteration += 1
        logger.join()
        return self._collapse_output() # return output


# solver that uses simulated annealing to solve the wave function collapse generation

class SimulatedAnnealingSolver():
    def __init__(self, td:TileDataset, n_iterations=10000, output_size=(32, 32), temperature=10000):
        # extract information about the tilesize output shape & heuristics
        self.td = td # containes all information about the tiles, constraints, and form
        self.output_size = output_size # will be of length 2 if format is HW or HWC
        self.n_tiles = len(td)
        self.probs = self.td.counts / self.td.counts.sum()
        # possible directions & form specific attributes
        if self.td.form == 'HW' or self.td.form == 'HWC':
            self.directions = {'height': 'left','width': 'up'} # only these directions are neede for neighbor comparison
        elif self.td.form == 'HWD':
            self.directions = {'height': 'left','width': 'up', 'depth': 'front'}


        self.n_iterations = n_iterations # assign number of iterations for the annealing

        # 1. create initial subset of features by weighted choice from the tile dataset
        self.W = np.random.choice(self.n_tiles, size=self.output_size, p=self.probs)
        self.new_W = self.W.copy()
        self.score = 0
        self.old_score = 0
        self.C = 0.2 # annealing constant
        self.temperature = temperature
        self.max_score = 1
        for i in self.output_size:
            self.max_score *= i

    def perturb(self, inp):
        # 2. perturb the feature subset
        mask = np.random.choice([False, True], size=self.output_size, p=[1-self.C, self.C])
        inp[mask] = np.random.choice(self.n_tiles, size=mask.sum(), p=self.probs)
        return inp

    def objective(self):
        # 3. score the performance of the perturbation of W
        # check how many of the neighbor constraints are satisfied
        score = 0
        for i, key in enumerate(self.directions.values()):
            edge = self.W.shape[i]
            if i == 0:
                comparison_array = self.W.flatten()
            if i == 1:
                comparison_array = np.rollaxis(self.W, 1).flatten()
            if i == 2:
                comparison_array = np.rollaxis(self.W, 2).flatten()

            for j in range(0, len(comparison_array)-1): # TODO vectorize to make it faster
                if j % edge != 0:
                    # compare the two tiles
                    A = self.td.constraints[key][comparison_array[j]].astype(bool)
                    B = self.td.constraints[key][comparison_array[j+1]].astype(bool)
                    if np.any(np.bitwise_and(A, B)):
                        score += 1
        return score


    def solve(self):
        # 4. optimize the feature subset
        pbar = tqdm(range(self.n_iterations))

        for it in pbar:
            try:
                pbar.set_postfix({'score': self.score, 'temperature': t, 'mac': mac, 'old_score': self.old_score})
            except:
                pass
            self.new_W = self.perturb(self.W)
            self.score = self.objective()
         
            if self.score > self.old_score:
                self.old_score = self.score
                self.W = self.new_W
            else:
                # calculate the probability of accepting the worse solution
                difference = (self.score - self.old_score)
                t = self.temperature / (it + 1)
                mac = 1 / np.exp(-difference / t)
                pbar.set_postfix({'score': self.score, 'temperature': t, 'mac': mac, 'old_score': self.old_score})
                if difference > 0 or np.random.uniform() < mac: # accept the worse solution
                    self.old_score = self.score
                    self.W = self.new_W

        return self._collapse_output() # return output

    def _collapse_output(self):
            out = np.zeros(self.output_size, dtype=int)
            if self.td.form == 'HWC':
                out = np.zeros(self.output_size + (4,), dtype=int) #change later this hardcodes four channels
            
            if self.td.form == 'HW' or self.td.form == 'HWC':
                for index in np.ndindex(self.output_size):
                    out[index] = self.td.cubes[self.W[index], 0, 0]
            
            elif self.td.form == 'HWD':
                for index in np.ndindex(self.output_size):
                    out[index] = self.td.cubes[self.W[index], 0, 0, 0]
            return out








