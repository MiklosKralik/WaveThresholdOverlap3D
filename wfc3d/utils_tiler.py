# functions for manipulating tiles
import numpy as np
from tqdm import tqdm
import os

def get_pad(x, N):
    return (N - x) % N

def pad_cubes(bitcube, pad_tup, fill,  wrap_pad):
    fill_tup = ((fill,), (fill,),(fill,))
    if not wrap_pad:
        print(f'padded with {fill}')
        H, W, D = bitcube.shape
        bitcube_fill = np.full((H+pad_tup[0][0],W+pad_tup[1][0],D+pad_tup[2][0]), fill)
        bitcube_fill[:H, :W, :D] = bitcube
        return bitcube_fill
    elif wrap_pad:
        print('wrapped')
        bitcube = np.pad(bitcube, pad_tup, mode='wrap')
        return bitcube

def reshape_window(bitcube, N, H_ratio, W_ratio, D_ratio):
    cubes = bitcube.reshape(H_ratio, N, W_ratio, N, D_ratio, N)
    cubes = np.moveaxis(cubes, [0,1,2,3,4,5], [0,2,4,1,3,5])
    return cubes.reshape(-1, N, N, N)

def find_fill_cubes(cubes):
    # returns indexes to keep
    condition = np.isnan(cubes)
    return np.any(condition, axis=(1, 2, 3)) == False


def extract_cubes(bitcube, N, wrap_pad=False):
    #extract attr
    H, W, D = bitcube.shape
    print(f'bitcube_shape: {bitcube.shape}')
    pad_H = get_pad(H, N)
    pad_W = get_pad(W, N)
    pad_D = get_pad(D, N)
    # pad if needed
    idx_org = np.arange(H*W*D).reshape(H, W, D) # for indexing original cube
    fill = np.nan
    pad_tup = ((pad_H, 0), (pad_W, 0), (pad_D, 0))
    bitcube_pad = pad_cubes(bitcube, pad_tup, fill, wrap_pad)
    idx_pad = pad_cubes(idx_org, pad_tup, fill, wrap_pad)
    print(f'bitcube_pad_shape: {bitcube_pad.shape}')
    # make sure correct dtype
    H_new, W_new, D_new = bitcube_pad.shape
    H_ratio = H_new // N
    W_ratio = W_new // N
    D_ratio = D_new // N
    total_cubes = H_new*W_new*D_new 
    cube_master = np.zeros((total_cubes, N, N, N))
    idx_master = np.zeros(cube_master.shape)
    num_cubes_per_iter = H_ratio*W_ratio*D_ratio
    master_shift = 0

    for H_shift in range(N):
        for W_shift in range(N):
            for D_shift in range(N):
                roll_tup = (H_shift, W_shift, D_shift)

                rolled_arr = np.roll(bitcube_pad, roll_tup, axis=(0, 1, 2))
                cube_master[master_shift*num_cubes_per_iter:(master_shift+1)*num_cubes_per_iter] = \
                    reshape_window(rolled_arr, N, H_ratio, W_ratio, D_ratio)
                
                idx_roll = np.roll(idx_pad, roll_tup, axis=(0, 1, 2))
                idx_master[master_shift*num_cubes_per_iter:(master_shift+1)*num_cubes_per_iter] = \
                    reshape_window(idx_roll, N, H_ratio, W_ratio, D_ratio)
                
                master_shift += 1
    keep_idxs = find_fill_cubes(cube_master)
    cube_master = cube_master[keep_idxs]
    idx_master = idx_master[keep_idxs]
    print(f'cube_master_dtype: {cube_master.dtype}')
    print(f'total cubes: {cube_master.shape}')
    return bitcube_pad, cube_master, idx_master

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

