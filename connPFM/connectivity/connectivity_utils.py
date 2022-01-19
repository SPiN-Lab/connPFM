"""Utility functions to perform event detection."""
import logging

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import zscore

LGR = logging.getLogger(__name__)


def calculate_ets(y, n):
    """Calculate edge-time series."""
    # upper triangle indices (node pairs = edges)
    u, v = np.argwhere(np.triu(np.ones(n), 1)).T

    # edge time series
    ets = y[:, u] * y[:, v]

    return ets, u, v


def rss_surr(z_ts, u, v, surrprefix, sursufix, masker, irand):
    """Calculate RSS on surrogate data."""
    [t, n] = z_ts.shape

    if surrprefix != "":
        zr = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
        if "AUC" not in surrprefix:
            zr = np.nan_to_num(zscore(zr, ddof=1))

        # TODO: find out why surrogates of AUC have NaNs after reading data with masker.
        zr = np.nan_to_num(zr)
    else:
        # perform numrand randomizations
        zr = np.copy(z_ts)
        for i in range(n):
            zr[:, i] = np.roll(zr[:, i], np.random.randint(t))

    # edge time series with circshift data
    etsr = zr[:, u] * zr[:, v]

    # calcuate rss
    rssr = np.sqrt(np.sum(np.square(etsr), axis=1))

    return (rssr, etsr, np.min(etsr), np.max(etsr))


def remove_neighboring_peaks(rss, idx):
    """
    Identify contiguous peaks among selected points in the RSS vector.

    Parameters
    ----------
    rss : ndarray
        RSS vector.
    idx : ndarray
        Indices of the selected peaks.

    Returns
    -------
    idxpeak: ndarray
        Indices of the selected peaks with no neighboring points.
    """
    # identify contiguous segments of frames that pass statistical test
    dff = idx.T - range(len(idx))
    unq = np.unique(dff)
    nevents = len(unq)

    # find the peak rss within each segment
    idxpeak = np.zeros([nevents, 1])
    for ievent in range(nevents):
        idxevent = idx[dff.T == unq[ievent].T]
        rssevent = rss[idxevent]
        idxmax = np.argmax(rssevent)
        idxpeak[ievent] = idxevent[idxmax]
    idxpeak = idxpeak[:, 0].astype(int)
    return idxpeak


def threshold_ets_matrix(ets_matrix, thr, selected_idxs=None):
    """
    Threshold the edge time-series matrix based on the selected time-points and
    the surrogate matrices.
    """
    # Initialize matrix with zeros
    thresholded_matrix = np.zeros(ets_matrix.shape)

    # Get selected columns from ETS matrix
    if selected_idxs is not None:
        thresholded_matrix[selected_idxs, :] = ets_matrix[selected_idxs, :]
    else:
        thresholded_matrix = ets_matrix

    # Threshold ETS matrix based on surrogate percentile
    # if thr is not an array, subtract it from the matrix
    if type(thr) is not np.ndarray:
        thresholded_matrix = thresholded_matrix - thr
    else:
        thresholded_matrix -= thr[:, None]

    thresholded_matrix[thresholded_matrix < 0] = 0

    return thresholded_matrix


def calculate_surrogate_ets(surrprefix, sursufix, irand, masker):
    """Read surrogate data."""
    auc = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
    [t, n] = auc.shape
    ets, _, _ = calculate_ets(np.nan_to_num(auc), n)

    return ets


def calculate_hist(
    surrprefix,
    sursufix,
    irand,
    masker,
    hist_range,
    nbins=500,
):
    """Calculate histogram."""
    ets_temp = calculate_surrogate_ets(surrprefix, sursufix, irand, masker)

    ets_hist, bin_edges = np.histogram(ets_temp.flatten(), bins=nbins, range=hist_range)

    return (ets_hist, bin_edges)


def calculate_hist_threshold(hist, bins, percentile=95):
    """Calculate histogram threshold."""
    ets_hist_sum = np.sum(hist, axis=0)
    cumsum_percentile = np.cumsum(ets_hist_sum) / np.sum(ets_hist_sum) * 100
    thr = bins[len(cumsum_percentile[cumsum_percentile <= percentile])]

    return thr


def surrogates_histogram(
    surrprefix,
    sursufix,
    masker,
    hist_range,
    numrand=100,
    nbins=500,
    percentile=95,
    all_hist=True,
):
    """
    Read AUCs of surrogates, calculate histogram and sum of all histograms to
    obtain a single histogram that summarizes the data.
    """

    if all_hist:
        ets_hist = np.zeros((numrand, nbins))

        # calculate histogram for each surrogate
        hist = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(calculate_hist)(surrprefix, sursufix, irand, masker, hist_range, nbins)
            for irand in range(numrand)
        )

        for irand in range(numrand):
            ets_hist[irand, :] = hist[irand][0]

        bin_edges = hist[0][1]

        # calculate histogram threshold
        thr = calculate_hist_threshold(ets_hist, bin_edges, percentile)
    else:
        all_thr = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(surrogate_threshold)(
                surrprefix, sursufix, masker, hist_range, irand, nbins, percentile
            )
            for irand in range(numrand)
        )

        thr = np.mean(all_thr)

    return thr


def surrogate_threshold(
    surrprefix,
    sursufix,
    masker,
    hist_range,
    irand,
    nbins=500,
    percentile=95,
):
    """
    Read AUCs of surrogates, calculate histogram and sum of all histograms to
    obtain a single histogram that summarizes the data.
    """
    hist, bin_edges = calculate_hist(surrprefix, sursufix, irand, masker, hist_range, nbins)

    # calculate histogram threshold
    thr = calculate_hist_threshold(hist, bin_edges, percentile)

    return thr
