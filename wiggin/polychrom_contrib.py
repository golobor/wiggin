import glob, os, pathlib
import logging
import collections
import shelve

import numpy as np
import pandas as pd

#import h5py

import polychrom
import polychrom.polymer_analyses
import polychrom.hdf5_format


def make_log_int_bins(end, start=1, bins_decade=10):
    lstart = np.log10(start)
    lend = np.log10(end - 1) 
    num = int(np.round(np.ceil((lend - lstart) * bins_decade)))
    bins = np.unique(
        np.round(
            np.logspace(lstart, lend, num=max(num, 0))).astype(np.int64))
    assert bins[-1] == end - 1
    return bins


def get_idx_spans(folder): # super blocks?..
    blocksh5 = list(pathlib.Path(folder).glob("blocks*.h5"))
    idx_spans = [f.name.split("_")[-1].split(".")[0].split("-") for f in blocksh5]
    idx_spans = sorted([(int(r[0]), int(r[1])) for r in idx_spans])

    return idx_spans


def get_last_block_idx(folder):
    idx_spans = get_idx_spans(folder)
    if len(idx_spans) == 0:
        return None
    last_block_idx = max(r[1] for r in idx_spans)

    return last_block_idx


def get_abs_block_idx(folder, idx):
    if idx < 0:
        idx = get_last_block_idx(folder) + 1 + idx

    return idx


def fetch_block(folder, ind=-1, full_output=False):
    """
    A more generic function to fetch block number "ind" from a trajectory in a folder
    
    
    This function is useful both if you want to load both "old style" trajectories (block1.dat), 
    and "new style" trajectories ("blocks_1-50.h5")
    
    It will be used in files "show" 
    
    Parameters
    ----------
    
        folder: str, folder with a trajectory

        ind: str or int, number of a block to fetch 
        
        full_output: bool (default=False)
            If set to true, outputs a dict with positions, eP, eK, time etc. 
            if False, outputs just the conformation
            (relevant only for new-style URIs, so default is False) 
    
    Returns
    -------
        coords, Nx3 numpy array     
        
        if full_output==True, then dict with coords and metacoords; XYZ is under key "pos"
    """
    blocksh5 = list(pathlib.Path(folder).glob("blocks*.h5"))
    blocksdat = list(pathlib.Path(folder).glob("block*.dat"))
    ind = int(ind)
    if (len(blocksh5) > 0) and (len(blocksdat) > 0):
        raise ValueError("both .h5 and .dat files found in folder - exiting")
    if (len(blocksh5) == 0) and (len(blocksdat) == 0):
        raise ValueError("no blocks found")

    if len(blocksh5) > 0:
        idx_spans = [f.name.split("_")[-1].split(".")[0].split("-") for f in blocksh5]
        idx_spans = [(int(r[0]), int(r[1])) for r in idx_spans]
        last_idx = max(r[1] for r in idx_spans)
        
        ind = get_abs_block_idx(folder, ind)

        idx_in_span = [(lo <= ind) and (hi >= ind) for lo, hi in idx_spans]

        if not any(idx_in_span):
            raise ValueError(f"block {ind} not found in files")
        if idx_in_span.count(True) > 1:
            raise ValueError("Cannot find the file uniquely: names are wrong")

        pos = idx_in_span.index(True)

        logging.info(f'Loading {ind}-th block out of {last_idx}...')

        block = polychrom.hdf5_format.load_URI(blocksh5[pos].as_posix() + f"::{ind}")
        if not full_output:
            block = block["pos"]

    if len(blocksdat) > 0:
        block = polychrom.polymerutils.load(
            (pathlib.Path(folder) / f"block{ind}.dat").as_posix()
        )
    return block


def contact_vs_sep(coords, bins_decade=10, bins=None, contact_radius=1.1, ring=False):
    """
    Returns the P(s) statistics of a polymer, i.e. the average contact frequency 
    vs countour separation. Contact frequencies are averaged across ranges (bins)
    of countour separation.
    
    Parameters
    ----------
    coords : Nx3 array of ints/floats
        An array of coordinates of the polymer particles.
    bins : array.
        Bins to divide the total span of possible countour separations.
        Either `bins` or `bins_decade` must be provided.
    bins_decade : int.
        If provided, separation bins are generated automatically
        in the range of [1, N_PARTICLES - 1],
        such that bins edges are approximately equally spaced in 
        the log space, with approx. `bins_decade` bins per decade.
        Either `bins` or `bins_decade` must be provided.
    contact_radius : float
        Particles separated in 3D by less than `contact_radius` are
        considered to be in contact.
    ring : bool, optional
        If True, will calculate contacts for the ring

    Returns
    -------
    (bin_mids, contact_freqs, npairs_per_bin) where "mids" contains
    geometric means of bin start/end
    

    """
    coords = np.asarray(coords)
    N = coords.shape[0]
    if coords.shape[1] != 3:
        raise ValueError('coords must contain an 2-D array of particle coordinates in 3D')

    contacts = np.array(polychrom.polymer_analyses.calculate_contacts(coords, contact_radius))
    contour_seps = np.abs(contacts[:, 1] - contacts[:, 0])  

    if ring:
        mask = contour_seps > N // 2
        contour_seps[mask] = N - contour_seps[mask]

    bins = make_log_int_bins(N, bins_decade=bins_decade) if bins is None else np.array(bins)
    bins[0] = bins[0] - 1 ## include the minimal separation into the first bin
    contacts_per_bin = np.bincount(
        np.searchsorted(bins, contour_seps, side="left"), 
        minlength=len(bins))
    contacts_per_bin = contacts_per_bin[1:len(bins)] # ignore contacts outside of distance binds

    if ring:
        pairs_per_bin = np.diff(N * bins)
    else:
        pairs_per_bin = np.diff(N * bins + 0.5 * bins - 0.5 * (bins ** 2))

    contact_freqs = contacts_per_bin / pairs_per_bin

    bin_mids = np.asarray(
        [np.sqrt(lo * hi) for lo, hi in zip(bins[:-1], bins[1:])])

    return pd.DataFrame({
        'sep': bin_mids, 
        'contact_freq': contact_freqs, 
        'n_particle_pairs': pairs_per_bin,
        'min_sep': bins[:-1],
        'max_sep': bins[1:],
        }
    )


def gaussian_contact_vs_sep(
        coords, 
        bins_decade=10, 
        bins=None, 
        contact_radius=1.1, 
        ring=False,
        random_sigma=3.0,
        random_reps=10,
    ):

    if random_sigma is None:
        return contact_freq_vs_sep(coords, 
            bins_decade=bins_decade, 
            bins=bins, 
            contact_radius=contact_radius, 
            ring=ring)

    contact_freqs = 0
    for _ in range(random_reps):
        shifts = np.random.normal(scale = random_sigma, size=coords.shape)
        res = contact_vs_sep(coords + shifts, bins_decade=bins_decade, bins=bins, contact_radius=contact_radius, ring=ring)
        contact_freqs += res['contact_freq'] / random_reps

    res['contact_freq'] = contact_freqs

    return res



SCALING_CACHE_FILENAME = 'scalings.shlv'
def cached_contact_vs_sep(
        folder, 
        block_idx=-1,
        bins_decade=10, 
        bins=None, 
        contact_radius=1.1, 
        ring=False,
        random_sigma=None,
        random_reps=10,
        
    ):

    if random_sigma is None:
        random_n_reps = None

    block_idx = get_abs_block_idx(folder, block_idx)

    path = pathlib.Path(folder) / SCALING_CACHE_FILENAME
    #cache_f = h5py.File(path.as_posix(), mode='a')
    cache_f = shelve.open(path.as_posix(), 'c')

    key_dict = {}
    for k in [
            'block_idx',
            'bins',
            'bins_decade',
            'contact_radius',
            'ring',
            'random_sigma',
            'random_reps',
        ]:
        key_dict[k] = locals()[k]


    if isinstance(key_dict['bins'], collections.abc.Iterable):
        key_dict['bins'] = tuple(key_dict['bins'])

    #key = '_'.join([i for for kv in sorted(key_dict.items()) for i in kv])
    key = repr(tuple(sorted(key_dict.items())))

    if key in cache_f:
        return cache_f[key]

    coords = fetch_block(folder, block_idx)
    sc = gaussian_contact_vs_sep(
        coords, 
        bins_decade=bins_decade, 
        bins=bins, 
        contact_radius=contact_radius, 
        ring=ring,
        random_sigma=random_sigma,
        random_reps=random_reps
    )
    cache_f[key] = sc
    cache_f.close()

    return sc
        
    

