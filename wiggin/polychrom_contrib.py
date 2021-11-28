import pathlib
import logging
import collections
import shelve

import numpy as np
import pandas as pd

import numba

import polychrom
import polychrom.polymer_analyses
import polychrom.hdf5_format


def make_log_int_bins(end, start=1, bins_decade=10):
    lstart = np.log10(start)
    lend = np.log10(end)
    num = int(np.round(np.ceil((lend - lstart) * bins_decade)))
    bins = np.unique(
        np.round(np.logspace(lstart, lend, num=max(num, 0))).astype(np.int64)
    )
    assert bins[-1] == end
    return bins


def get_idx_spans(folder):  # super blocks?..
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
    A more generic function to fetch block number "ind" from a trajectory
    in a folder.

    This function is useful both if you want to load both "old style"
    trajectories (block1.dat), and "new style" trajectories ("blocks_1-50.h5")

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

        if full_output==True, then dict with coords and metacoords;
        XYZ is under key "pos"
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

        logging.info(f"Loading {ind}-th block out of {last_idx}...")

        block = polychrom.hdf5_format.load_URI(blocksh5[pos].as_posix() + f"::{ind}")
        if not full_output:
            block = block["pos"]

    if len(blocksdat) > 0:
        block = polychrom.polymerutils.load(
            (pathlib.Path(folder) / f"block{ind}.dat").as_posix()
        )
    return block


def _bin_contacts(contacts, N, bins_decade=10, bins=None, ring=False):
    if ring:
        contour_dists = np.abs(contacts[:, 1] - contacts[:, 0])
        mask = contour_dists > N // 2
        contour_dists[mask] = N - contour_dists[mask]
    else:
        contour_dists = np.abs(contacts[:, 1] - contacts[:, 0])

    if bins is None and bins_decade is None:
        bins = np.arange(0, N + 1)
        contacts_per_bin = np.bincount(contour_dists, minlength=N)
    else:
        if bins is None:
            bins = make_log_int_bins(N, bins_decade=bins_decade)
        else:
            bins = np.array(bins)

        contacts_per_bin = np.bincount(
            np.searchsorted(bins, contour_dists, side="right"), minlength=len(bins)
        )
        contacts_per_bin = contacts_per_bin[
            1 : len(bins)
        ]  # ignore contacts outside of distance binds

    if ring:
        pairs_per_bin = np.diff(N * bins)
    else:
        pairs_per_bin = np.diff(N * bins + 0.5 * bins - 0.5 * (bins ** 2))

    contact_freqs = contacts_per_bin / pairs_per_bin

    bin_mids = np.sqrt(bins[:-1] * bins[1:])

    return pd.DataFrame(
        {
            "dist": bin_mids,
            "contact_freq": contact_freqs,
            "n_particle_pairs": pairs_per_bin,
            "n_contacts": contacts_per_bin,
            "min_dist": bins[:-1],
            "max_dist": bins[1:],
        }
    )


def contact_vs_dist(
    coords,
    contact_radius=1.1,
    bins_decade=10,
    bins=None,
    ring=False,
):
    """
    Returns the P(s) statistics of a polymer, i.e. the average contact frequency
    vs countour distance. Contact frequencies are averaged across ranges (bins)
    of countour distance.

    Parameters
    ----------
    coords : Nx3 array of ints/floats
        An array of coordinates of the polymer particles.
    bins : array.
        Bins to divide the total span of possible countour distances.
        If neither `bins` or `bins_decade` are provided, distances are not binned.
    bins_decade : int.
        If provided, distance bins are generated automatically
        in the range of [1, N_PARTICLES - 1],
        such that bins edges are approximately equally spaced in
        the log space, with approx. `bins_decade` bins per decade.
        If neither `bins` or `bins_decade` are provided, distances are not binned.

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
    if coords.shape[1] != 3:
        raise ValueError(
            "coords must contain an Nx3 array of particle coordinates in 3D"
        )
    N = coords.shape[0]

    contacts = np.array(
        polychrom.polymer_analyses.calculate_contacts(coords, contact_radius)
    )

    assert np.sum(contacts[:, 1] < contacts[:, 0]) == 0

    cvd = _bin_contacts(contacts, N, bins_decade=10, bins=None, ring=False)

    return cvd


def is_in(arr, lo, hi):
    return (arr >= lo) & (arr < hi)


def contact_vs_dist_multichain(
    coords,
    chains,
    contact_radius=1.1,
    bins_decade=10,
    bins=None,
    ring=False,
    trans=True,
):

    chains = sorted(chains, key=lambda ch: ch[0])
    if all([0 <= i <= 1 for ch in chains for i in ch]):
        N = coords.shape[0]
        chains = [(int(np.round(ch[0] * N)), int(np.round(ch[1] * N))) for ch in chains]

    for i, chain1 in enumerate(chains):
        for chain2 in chains[i + 1 :]:
            if chain1[1] > chain2[0]:
                raise ValueError("Chains must not overlap!")

    chain_lens = [ch[1] - ch[0] for ch in chains]
    if trans and len(set(chain_lens)) > 1:
        raise ValueError(
            "Trans contacts can only be analysed for chains of equal length!"
        )

    contacts = np.array(
        polychrom.polymer_analyses.calculate_contacts(coords, contact_radius)
    )

    assert np.sum(contacts[:, 1] < contacts[:, 0]) == 0

    cvds = []
    cis_chain_pairs = [(ch, ch) for ch in chains]
    trans_chain_pairs = [
        (ch1, ch2) for i, ch1 in enumerate(chains) for ch2 in chains[i + 1 :]
    ]
    chain_pairs_to_calc = (
        cis_chain_pairs + trans_chain_pairs if trans else cis_chain_pairs
    )

    for chain1, chain2 in chain_pairs_to_calc:
        contact_mask = is_in(contacts[:, 0], *chain1) & is_in(contacts[:, 1], *chain2)
        loc_contacts = contacts[contact_mask]

        # This part can be done more elegantly:
        # in _bin_counts, factor out the calculation of countour separations and n_pairs
        # and instead provide a function that can calculate these values for every distance
        # (and then sum over bins - wasteful, but flexible)
        # create an extra function, which provides presents for these calculations for
        # rings, trans, etc...
        if chain1 != chain2:
            loc_contacts = np.vstack(
                [loc_contacts[:, 0] - chain1[0], loc_contacts[:, 1] - chain2[0]]
            ).T

        cvd = _bin_contacts(
            loc_contacts,
            chain1[1] - chain1[0],
            bins_decade=bins_decade,
            bins=bins,
            ring=ring,
        )

        if chain1 != chain2:
            cvd["n_particle_pairs"] *= 2
            cvd["contact_freq"] /= 2

        cvd["chain1"] = f"{chain1[0]}-{chain1[1]}"
        cvd["chain2"] = f"{chain2[0]}-{chain2[1]}"
        cvds.append(cvd)

    return pd.concat(cvds)


def gaussian_contact_vs_dist(
    coords, contact_vs_dist_func, random_sigma=3.0, random_reps=10, **kwargs
):

    if random_sigma is None:
        return contact_vs_dist_func(coords, **kwargs)

    contact_freqs = 0
    for _ in range(random_reps):
        shifts = np.random.normal(scale=random_sigma, size=coords.shape)
        res = contact_vs_dist_func(coords + shifts, **kwargs)
        contact_freqs += res["contact_freq"] / random_reps

    res["contact_freq"] = contact_freqs

    return res


SCALING_CACHE_FILENAME = "scalings.shlv"


def cached_contact_vs_dist(
    folder,
    block_idx=-1,
    contact_radius=1.1,
    bins_decade=10,
    bins=None,
    ring=False,
    random_sigma=None,
    random_reps=10,
    cache_file=SCALING_CACHE_FILENAME,
):

    if random_sigma is None:
        random_n_reps = None

    block_idx = get_abs_block_idx(folder, block_idx)

    path = pathlib.Path(folder) / SCALING_CACHE_FILENAME
    cache_f = shelve.open(path.as_posix(), "c")

    key_dict = {}
    for k in [
        "block_idx",
        "bins",
        "bins_decade",
        "contact_radius",
        "ring",
        "random_sigma",
        "random_reps",
    ]:
        key_dict[k] = locals()[k]

    if isinstance(key_dict["bins"], collections.abc.Iterable):
        key_dict["bins"] = tuple(key_dict["bins"])

    # key = '_'.join([i for for kv in sorted(key_dict.items()) for i in kv])
    key = repr(tuple(sorted(key_dict.items())))

    if key in cache_f:
        return cache_f[key]

    coords = fetch_block(folder, block_idx)
    sc = gaussian_contact_vs_dist(
        coords,
        contact_vs_dist_func=contact_vs_dist,
        random_sigma=random_sigma,
        random_reps=random_reps,
        bins_decade=bins_decade,
        bins=bins,
        contact_radius=contact_radius,
        ring=ring,
    )
    cache_f[key] = sc
    cache_f.close()

    return sc


@numba.njit
def log_thin(xs, min_log10_step=0.1):
    xs_thinned = [xs[0]]
    prev = xs[0]
    min_ratio = 10 ** min_log10_step
    for x in xs[1:]:
        if x > prev * min_ratio:
            xs_thinned.append(x)
            prev = x

    if xs_thinned[-1] != xs[-1]:
        xs_thinned.append(xs[-1])
    return np.array(xs_thinned)


def log_interp(xs, xp, fp):
    return np.exp(
        np.interp(
            np.log(xs),
            np.log(xp),
            np.log(fp),
        )
    )


@numba.jit(nopython=True)
def log_smooth_ratio(
    xs,
    y_numerators,
    y_denominators,
    sigma_log10=0.1,
    window_sigma=5,
    steps_per_sigma=10,
):
    assert xs.ndim == 1
    assert xs.shape == y_numerators.shape == y_denominators.shape

    thinned_xs = xs
    if steps_per_sigma:
        thinned_xs = log_thin(xs, sigma_log10 / steps_per_sigma)

    N = thinned_xs.size

    log_xs = np.log10(xs)
    log_thinned_xs = np.log10(thinned_xs)

    y_numerators_smooth = np.zeros(N)  # , dtype=np.float)
    y_denominators_smooth = np.zeros(N)  # , dtype=np.float)

    for i in range(N):
        cur_log_x = log_thinned_xs[i]
        lo = np.searchsorted(log_xs, cur_log_x - sigma_log10 * window_sigma)
        hi = np.searchsorted(log_xs, cur_log_x + sigma_log10 * window_sigma)
        smooth_weights = np.exp(
            -((cur_log_x - log_xs[lo:hi]) ** 2) / 2 / sigma_log10 / sigma_log10
        )
        y_numerators_smooth[i] = np.sum(y_numerators[lo:hi] * smooth_weights)
        y_denominators_smooth[i] = np.sum(y_denominators[lo:hi] * smooth_weights)

    return thinned_xs, y_numerators_smooth, y_denominators_smooth


def agg_smooth_cvd(cvd, sigma_log10=0.1, window_sigma=5, steps_per_sigma=10, **kwargs):

    dist_col = kwargs.get("dist_col", "dist")
    n_pairs_col = kwargs.get("n_pairs_col", "n_particle_pairs")
    n_contacts_col = kwargs.get("n_contacts_col", "n_contacts")
    contact_freq_col = kwargs.get("contact_freq_col", "contact_freq")
    smooth_suffix = kwargs.get("smooth_suffix", ".smoothed")

    cvd_agg = (
        cvd.groupby(dist_col)
        .agg(
            {
                n_pairs_col: "sum",
                n_contacts_col: "sum",
            }
        )
        .reset_index()
    )

    bin_mids, balanced, areas = log_smooth_ratio(
        cvd_agg[dist_col].values.astype(np.float64),
        cvd_agg[n_contacts_col].values.astype(np.float64),
        cvd_agg[n_pairs_col].values.astype(np.float64),
        sigma_log10=sigma_log10,
        steps_per_sigma=steps_per_sigma,
    )

    if steps_per_sigma:
        cvd_agg[n_pairs_col + smooth_suffix] = log_interp(
            cvd_agg[dist_col].values, bin_mids, areas
        )
        cvd_agg[n_contacts_col + smooth_suffix] = log_interp(
            cvd_agg[dist_col].values, bin_mids, balanced
        )
    else:
        cvd_agg[n_pairs_col + smooth_suffix] = areas
        cvd_agg[n_contacts_col + smooth_suffix] = balanced

    cvd_agg[contact_freq_col] = cvd_agg[n_contacts_col] / cvd_agg[n_pairs_col]
    cvd_agg[contact_freq_col + smooth_suffix] = (
        cvd_agg[n_contacts_col + smooth_suffix] / cvd_agg[n_pairs_col + smooth_suffix]
    )

    return cvd_agg
