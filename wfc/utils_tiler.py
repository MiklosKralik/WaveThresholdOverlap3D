# functions for manipulating tiles
import numpy as np
from tqdm import tqdm
import os

# Requirements for cube extraction:
# Functions to extact all overlapping tiles or cubes from a numpy array
# bitcube input may have the following formats: HW, HWC, HWD
# N is the size of the tile or cube
# O is the overlap, which is always N-1
# if wrap_pad is True the the array is padded with wrap around otherwise edge tiles will be thrown away

def get_pad(x, N):
    return (N - x) % N

def get_pad_shape(shape, N):
    pad_shapes = []
    for s in shape:
        pad_shapes.append(get_pad(s, N))
    return tuple(pad_shapes)

def pad(bitcube, wrap_pad, pad_vals):
    # pad_vals is a tuple of dims to be padded with values
    # wrap pad is bool
    pad_vals = tuple([(0, p) for p in pad_vals])
    if wrap_pad:
        return np.pad(bitcube, pad_vals, mode='wrap')
    elif not wrap_pad: # pad with nans
        return np.pad(bitcube, pad_vals, mode='constant', constant_values=np.nan)

def window(bitcube, form, N):
    # get all overlapping cubes from bitcube
    if form == 'HW':
        H, W = bitcube.shape
        H_ratio = H//N
        W_ratio = W//N
        cubes = bitcube.reshape(H_ratio, N, W_ratio, N)
        cubes = np.moveaxis(cubes, [0,1,2,3], [0,2,1,3])
        cubes = cubes.reshape(-1, N, N)
    elif form == 'HWC':
        H, W, C = bitcube.shape
        H_ratio = H//N
        W_ratio = W//N
        cubes = bitcube.reshape(H_ratio, N, W_ratio, N, C)
        cubes = np.moveaxis(cubes, [0,1,2,3,4], [0,2,1,3,4])
        cubes = cubes.reshape(-1, N, N, C)
    elif form == 'HWD':
        H, W, D = bitcube.shape
        H_ratio = H//N
        W_ratio = W//N
        D_ratio = D//N
        cubes = bitcube.reshape(H_ratio, N, W_ratio, N, D_ratio, N)
        cubes = np.moveaxis(cubes, [0,1,2,3,4,5], [0,2,4,1,3,5])
        cubes = cubes.reshape(-1, N, N, N)
    return cubes

def construct_master(bitcube, form, N):
    # construct master array to store cubes
    # roll array and window
    if form == 'HW':
        H, W = bitcube.shape
        master = np.empty((H*W, N, N), dtype=bitcube.dtype)
        for i in range(N):
            for j in range(N):
                master[(H*W)//N**2*(i+j):(H*W)//N**2*(i+j+1)] = window(np.roll(bitcube, (i, j), axis=(0, 1)), form, N)
    elif form == 'HWC':
        H, W, C = bitcube.shape
        master = np.empty((H*W, N, N, C), dtype=bitcube.dtype)
        for i in range(N):
            for j in range(N):
                master[(H*W)//N**2*(i+j):(H*W)//N**2*(i+j+1)] = window(np.roll(bitcube, (i, j), axis=(0, 1)), form, N)
    elif form == 'HWD':
        H, W, D = bitcube.shape
        master = np.empty((H*W*D, N, N, N), dtype=bitcube.dtype)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    master[(H*W*D)//N**3*(i+j+k):(H*W*D)//N**3*(i+j+k+1)] = window(np.roll(bitcube, (i, j, k), axis=(0, 1, 2)), form, N)
    return master


def extract_cubes(bitcube, form, N=2, wrap_pad=False):
    """ Extracts all overlapping cubes from a numpy array
    Args:
    bitcube input may have the following forms: HW, HWC, HWD
    bitcube: numpy array of type int
    N is the size of the tile or cube
    wrap_pad is True the the array is padded with wrap around otherwise edge tiles will be thrown away """
    # assert bicube is of type int
    try:
        assert bitcube.dtype == np.int
    except AssertionError:
        print('bitcube is not of type int')
        return
    O = N-1

    if form == 'HW':
        H, W = bitcube.shape
        H_pad, W_pad = get_pad_shape(bitcube.shape, N)
        bitcube = pad(bitcube, wrap_pad, (H_pad, W_pad))

    elif form == 'HWC':
        H, W, C = bitcube.shape
        H_pad, W_pad = get_pad_shape(bitcube.shape[-1], N)
        C = bitcube.shape[-1]
        bitcube = pad(bitcube, wrap_pad, (H_pad, W_pad))

    elif form == 'HWD':
        H, W, D = bitcube.shape
        H_pad, W_pad, D_pad = get_pad_shape(bitcube.shape, N)
        bitcube = pad(bitcube, wrap_pad, (H_pad, W_pad, D_pad))
    
    idx_master = None
    print(bitcube.shape)
    cubes = construct_master(bitcube, form, N)
    return bitcube, cubes, idx_master


def find_fill_cubes(cubes):
    # returns indexes to keep
    condition = np.isnan(cubes)
    return np.any(condition, axis=(1, 2, 3)) == False

def return_shift(D, N, O):
    H_shift = (0, N)
    W_shift = (0, N)
    D_shift = (0, N)
    shift = (0, O)
    opp_shift = (N-O, N)
    if D == 'left':
        W_shift = shift
    elif D == 'right':
        W_shift = opp_shift
    elif D == 'up':
        H_shift = shift
    elif D == 'down':
        H_shift = opp_shift
    elif D == 'forwards':
        D_shift = shift
    elif D == 'backwards':
        D_shift = opp_shift
    return H_shift, W_shift, D_shift

def compare_cubes(cube_arr, O, one_dir=False):
    N = cube_arr.shape[-1]
    l = cube_arr.shape[0]
    opp_dict = {'left':'right', 'right':'left', 'up':'down', 'down':'up', 'forwards':'backwards', 'backwards':'forwards'}
    pairwise_master = {'left':np.empty((l, l), dtype='bool'), 'right':np.empty((l, l), dtype='bool'), 
                        'up':np.empty((l, l), dtype='bool'), 'down':np.empty((l, l), dtype='bool'),
                        'forwards':np.empty((l, l), dtype='bool'), 'backwards':np.empty((l, l), dtype='bool')}
    if one_dir:
        print('only comparing in one dir')
        opp_dict = {'left':'right'}
        pairwise_master = {'left':np.empty((l, l), dtype='bool')}

    for D in opp_dict.keys():
        H_all, W_all, D_all = return_shift(D, N, O)
        all_cubes_O = cube_arr[:, H_all[0]:H_all[1], W_all[0]:W_all[1], D_all[0]:D_all[1]]
        H_s, W_s, D_s = return_shift(opp_dict[D], N, O)
        s_cubes_O = cube_arr[:, H_s[0]:H_s[1], W_s[0]:W_s[1], D_s[0]:D_s[1]]
        dummy_ind = np.arange(N*O)
        unravel_tup = (H_all[1] - H_all[0], W_all[1] - W_all[0], D_all[1] - D_all[0])
        pixH, pixW, pixD = np.unravel_index(dummy_ind, unravel_tup) 
        unravel_tup = (H_s[1] - H_s[0], W_s[1] - W_s[0], D_s[1] - D_s[0])
        pixH_s, pixW_s, pixD_s = np.unravel_index(dummy_ind, unravel_tup)
        num_pix = len(pixH)

        for cube_idx, cube_O in tqdm(enumerate(s_cubes_O)):
            comp_bool = np.full(l, True)
            for i in range(num_pix):
                x = pixH[i]
                y = pixW[i]
                z = pixD[i]
                x_s = pixH_s[i]
                y_s = pixW_s[i]
                z_s = pixD_s[i]
                comp_ind = np.where(np.ma.masked_array(all_cubes_O[:, x, y, z], mask=(comp_bool==False)) \
                        != cube_O[x_s, y_s, z_s])
                comp_bool[comp_ind] = False
            pairwise_master[D][cube_idx] = comp_bool
    return pairwise_master

def svd(constraints):
    return np.linalg.svd(constraints)

def threshold_singular(u, s, vh, threshold):
    take_idxs = (s >= threshold)
    U = np.zeros(u.shape)
    S = np.zeros(u.shape)
    Vh = np.zeros(vh.shape)
    U[:, take_idxs] = u[:, take_idxs]
    S[np.diag_indices(len(s))] = s
    Vh[take_idxs] = vh[take_idxs]
    connect_new = U@S@Vh
    keep_cubes = (connect_new.sum(axis=0) >=1)
    return keep_cubes

def rotate_cubes(cube_arr):
    """ rotate cubes in 6 directions and concatenate them along axis 0"""
    axis_tup = ((1, 2), (1, 3), (2, 3))
    rot_k = [0, 1, 2, 3]
    cube_arr_list = []
    for axes in axis_tup:
        for k in rot_k:
            cube_arr_list.append(np.rot90(cube_arr, k=k, axes=axes))

    return np.concatenate(cube_arr_list, axis=0)


def flip_cubes(cube_arr):
    cube_arr_list = [cube_arr, np.flip(cube_arr, axis=1), np.flip(cube_arr, axis=2)]
    return np.concatenate(cube_arr_list, axis=0)

def normalize_cube(cube):
    """ normalize_cubes for better viewing"""
    cube = cube.swapaxes(0, 2)
    cube = cube / cube.max()
    cube = (cube * 255).astype(np.uint8)
    return cube

def rot_flip_unique(cube_arr):
    """ collate rotate and flip """
    cube_arr = flip_cubes(rotate_cubes(cube_arr))
    cubes_unique, index, counts = np.unique(cube_arr,
            axis=0, return_counts=True, return_index=True)
    print(f'rot_flip_unique: {cubes_unique.shape}')
    return cubes_unique, counts, index

def td_rot_flip(td):
    td.cubes, td.counts, index  = rot_flip_unique(td.cubes)
    td.constraints = compare_cubes(td.cubes, td.O)
    if td.raw is not None:
        td.raw_cubes = flip_cubes(rotate_cubes(td.raw_cubes))[index]
    return td

def td_rot_flip_threshold(td, threshold=0, view_svd=False, use_jax=False, rotate_flip=True):
    """ preforms svd thresholding on td dataset class"""
    if rotate_flip:
        td.cubes, td.counts = rot_flip_unique(td.cubes)
        td.constraints = compare_cubes(td.cubes, td.O)
    keep_list = []
    for k in tqdm(td.constraints.keys()):
        if use_jax:
            u, s, vh = jnp.linalg.svd(td.constraints[k])
        else:
            u, s, vh = np.linalg.svd(td.constraints[k])
        if view_svd: # for plotting
            return s
        keep_list.append(threshold_singular(u, s, vh, threshold))
    keep_cubes = np.vstack(keep_list).all(axis=0)
    print(f' thresholded_cubes: {keep_cubes[keep_cubes].shape}')
    td.cubes = td.cubes[keep_cubes]
    td.counts = td.counts[keep_cubes]
    for k in td.constraints.keys():
        td.constraints[k] = td.constraints[k][keep_cubes][:, keep_cubes]
    return td

