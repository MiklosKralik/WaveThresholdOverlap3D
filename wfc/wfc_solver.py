# imports
import numpy as np
import sys
from wfc.wfc_tiler import TileDataset
from scipy.stats import entropy
import time
import threading

class WaveSolver:
    def __init__(self, td:TileDataset, output_size:tuple):
        
        # extract information about the tilesize output shape & heuristics
        self.td = td # containes all information about the tiles, constraints, and form
        self.output_size = output_size # will be of length 2 if format is HW or HWC
        self.n_tiles = len(td)
        self.mode = 'wrap' # constraints will wrap around output else raise
        self.max_contra = 100

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
            log_prop = f'\033[91m PROPAGATION:\033[91m {self.prop:10}'
            log_iter = f'\033[35m ITER:\033[35m {self.iteration:10}'
            log_status = f'\033[36m STATUS: \033[36m' + self.status
            log_S = f'\033[95m STACK:\033[95m {len(self.S):10}'
            log_collapsed = f'COLL: {self.collapsed:8}'
            sys.stdout.write('\r')
            sys.stdout.write(log_prop)
            sys.stdout.write(log_iter)
            sys.stdout.write(log_S)
            sys.stdout.write(log_status)
            sys.stdout.write(log_collapsed)
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
#            self._backtrack('record')
            self.S.append(coords)
            self._propagate()
#            if self.status == 'FAIL' and self.contra < self.max_contra:
#                self._backtrack('reset')
#                self.contra += 1
#                self.status = 'RUN'
            self.status = self._check_for_collapse()
            if self.status not in ['FAIL', 'COLLAPSE']:
                coords = self._observe()
            self.iteration += 1
        logger.join()
        return self._collapse_output() # return output



#class WaveSolver:
#
#    def __init__(self, td:TileDataset, output_size:tuple, log_time=1, mode='raise'):
#
#        # initialize wave and entropy matrix
#        self.td = td
#        self.l = len(td.cubes)
#        self.output_size = output_size
#        self.height, self.width, self.depth = output_size
#        self.W = np.full((self.height, self.width, self.depth, self.l), True, dtype=bool)
#        self.H = entropy(self.W, axis=3).astype(np.float16)
#        self.S = [] # list to store propagation keys
#        self.status = 'INIT' # can be RUN FAIL COLLAPSE
#        self.directions = ['left', 'right', 'up', 'down', 'front', 'back']
#        self.dir_trans = {'left':(-1, 0, 0), 'right':(1, 0, 0), 'up':(0, 1, 0),
#                'down':(0, -1, 0), 'back':(0, 0, -1),
#                'front':(0, 0, 1)}# make an dict to find neighbor indicies
#        self.mode=mode # wrap constraints 'wrap' else 'raise'
#        # metrics
#        self.prop = 0
#        self.iteration = 0
#        self.contra = 0
#        self.collapsed = 0
#        self.max_contra = 10000
#        self.log_time = log_time
#        # backtracking
#        self.last_W = self.W
#        self.last_S = self.S
#        self.last_H = self.H
#        
#        #output
#        self.out = False
#    
#    def _update_entropy(self):
#        # update entropy based on weight matrix
#        self.H = entropy(self.W, axis=3) # may be worth doing element-wise
#        
#    def _random_cell_choice(self):
#        # find minimum entropy cells make a random choice and return coords
#        min_idxs = np.where(self.H == self.H[np.nonzero(self.H)].min())
#        i = np.random.randint(len(min_idxs[0]))
#        return min_idxs[0][i], min_idxs[1][i], min_idxs[2][i]
#
#    def _random_cube_collapse(self, coords):
#        # picks a random cube idx based on frequency probabilities and collapses it
#        available_idxs = np.argwhere(self.W[coords])[:, 0]
#        probs = self.td.counts[available_idxs]
#        choice = np.random.choice(available_idxs, p=probs / probs.sum())
#        self.W[coords] = False
#        self.W[coords][choice] = True
#    
#    def _check_cell_for_collapse(self, cubes_bool):
#        if np.count_nonzero(cubes_bool) == 1:
#            return True
#        else:
#            return False
#
#    def _check_for_collapse(self):
#        # check if all cells have collapsed, do not check if self.status is FAIL
#        if self.status == 'FAIL':
#            return 'FAIL'
#        collapsed = np.count_nonzero(self.W == True, axis=3) == 1
#        self.collapsed = collapsed.sum()
#        if np.all(collapsed):
#            return 'COLLAPSE'
#        else:
#            return 'RUN'
#
#    def _apply_neighbors(self, coords, trans):
#        return coords[0] + trans[0], coords[1] + trans[1], coords[2] + trans[2]
#
#    def _find_neighbors(self, coords):
#        # find neighbors of a cell from and return a list of valid ones
#        neighbors = {}
#        for d in self.directions:
#            ind = self._apply_neighbors(coords, self.dir_trans[d])
#            try:
#                flat_index = np.ravel_multi_index(ind, self.output_size, mode=self.mode)
#                neighbors[d] = np.unravel_index(flat_index, self.output_size)
#            except ValueError:
#                neighbors[d] = None
#        return neighbors
#
#    def _backtrack(self, state):
#        if state == 'record':            
#            self.last_W = self.W
#            self.last_H = self.H
#        elif state == 'reset':
#            self.W = self.last_W
#            self.S = [] # stack must be emptied
#            self.H = self.last_H
#
#    def _observe(self):
#        # observe wave matrix, collapse cell with the lowest entropy and return coords
#        self._update_entropy()
#        coords = self._random_cell_choice()
#        self._random_cube_collapse(coords)
#        return coords
#
#    def _propagate(self):
#        self.prop = 0
#        while len(self.S) > 0:
#            coords = self.S.pop(-1)
#            neighbors = self._find_neighbors(coords)
#            source_cubes = self.W[coords]
#            for direction, n_coords in neighbors.items():
#                # check if direction is available
#                if n_coords == None:
#                    continue
#                destination_cubes = self.W[n_coords]
#                # if not collapsed and if not update == contradiction and if not update = destination
#                # add to stack
#                if self._check_cell_for_collapse(destination_cubes):
#                    continue
#                adj_cubes = self.td.constraints[direction][source_cubes].any(axis=0)
#                n_update = np.bitwise_and(destination_cubes, adj_cubes)
#                if np.all(n_update == False):
#                    self.status = 'FAIL'
#                    return 'FAIL'
#                if (n_update == destination_cubes).all():
#                    continue
#
#                self.W[n_coords] = n_update
#                self.S.append(n_coords)
#                self.prop += 1
#
#    def _collapse_output(self):
#        out = np.full(self.output_size, 0)
#        collapsed_W = self.W.argmax(axis=3)
#        for index in np.ndindex(self.output_size):
#            out[index] = self.td.cubes[collapsed_W[index], 0, 0, 0]
#        self.out = out
#        return out
#
#    def _log(self):
#        log_count = 1
#        while self.status not in ['FAIL', 'COLLAPSE']:
#            last_prop = self.prop
#            time.sleep(self.log_time)
#            log_count += 1
#            log_prop = f'\033[91m PROPAGATION:\033[91m {self.prop:10}'
#            log_iter = f'\033[35m ITER:\033[35m {self.iteration:10}'
#            log_status = f'\033[36m STATUS: \033[36m' + self.status
#            log_S = f'\033[95m STACK:\033[95m {len(self.S):10}'
#            log_contra = f'\033[34m CONTRA: {self.contra:8}'
#            log_collapsed = f'COLL: {self.collapsed:8}'
#            sys.stdout.write('\r')
#            sys.stdout.write(log_prop)
#            sys.stdout.write(log_iter)
#            sys.stdout.write(log_S)
#            sys.stdout.write(log_contra)
#            sys.stdout.write(log_status)
#            sys.stdout.write(log_collapsed)
#            sys.stdout.flush()
#                        
#
#    def save_output(self, path):
#        out, out_raw  = self._collapse_output()
#        np.save(path, out)
#        np.save(path + '_raw', out_raw)
#
#    def solve(self):
#        logger = threading.Thread(target=self._log)
#        logger.start()
#        print('solving...')
#        self.status = self._check_for_collapse()
#        self.iteration = 0
#        coords = self._observe() # cell to start prop from
#        while self.status not in ['FAIL', 'COLLAPSE']:
#            self._backtrack('record')
#            self.S.append(coords)
#            self._propagate()
#            if self.status == 'FAIL' and self.contra < self.max_contra:
#                self._backtrack('reset')
#                self.contra += 1
#                self.status = 'RUN'
#            coords = self._observe()
#            self.status = self._check_for_collapse()
#            self.iteration += 1
