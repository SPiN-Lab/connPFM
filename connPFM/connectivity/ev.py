"""Main workflow to perform event detection."""
import logging
from os.path import join

import numpy as np
from dask import compute
from dask import delayed as delayed_dask
from scipy.sparse import save_npz

from connPFM.connectivity import connectivity_utils
from connPFM.connectivity.plotting import plot_ets_matrix
from connPFM.utils.io import dask_scheduler, load_data

LGR = logging.getLogger(__name__)


def event_detection(
    data_file,
    atlas,
    surrprefix="",
    sursufix="",
    nsur=100,
    segments=True,
    peak_detection="rss",
    nbins=1000,
    te=[0],
    jobs=1,
):
    """Perform event detection on given data."""
    auc_ts, masker = load_data(data_file, atlas, n_echos=len(te))
    # Get number of time points/nodes
    [t, n] = auc_ts.shape

    # calculate ets
    ets, u, v = connectivity_utils.calculate_ets(auc_ts, n)

    # Initialize thresholded edge time-series matrix with zeros
    etspeaks = np.zeros(ets.shape)

    # Initialize array of indices of detected peaks
    idxpeak = np.zeros(t)

    # Initialize RSS with zeros
    rss = np.zeros(t)
    rss_th = None
    # Initialize array for null RSS
    rssr = np.zeros([t, nsur])

    # calculate ets and rss of surrogate data
    LGR.info("Calculating edge-time matrix, RSS and histograms for surrogates...")
    _, cluster = dask_scheduler(jobs)
    futures = []
    for irand in range(nsur):
        fut = delayed_dask(connectivity_utils.rss_surr, pure=False)(
            auc_ts, u, v, surrprefix, sursufix, masker, irand, nbins
        )
        futures.append(fut)
    surrogate_events = compute(futures)[0]
    # Make selection of points with RSS
    if "rss" in peak_detection:
        LGR.info("Selecting points with RSS...")
        # calculate rss
        rss = np.array(np.sqrt(ets.power(2).sum(axis=1)[:, 0].flatten())).flatten()

        for irand in range(nsur):
            rssr[:, irand] = surrogate_events[irand][0]

        # TODO: find out why there is such a big peak on time-point 0 for AUC surrogates
        if "AUC" in surrprefix:
            rssr[0, :] = 0

        # Initialize p-values array with ones to avoid default values being lower
        # than the threshold p-value
        p_value = np.ones(t)

        # Statistical cutoff
        pcrit = 0.001

        # Find peaks with a general threshold for RSS
        if peak_detection == "rss":
            rssr_flat = rssr.flatten()
            for i in range(t):
                p_value[i] = np.mean(rssr_flat >= rss[i])

        # Find peaks with a different threshold for each time-point in RSS
        elif peak_detection == "rss_time":
            for i in range(t):
                p_value[i] = np.mean(np.squeeze(rssr[i, :]) >= rss[i])

        # find frames that pass statistical test auc_ts
        idx = np.argwhere(p_value < pcrit)[:, 0]
        if segments:
            idxpeak = connectivity_utils.remove_neighboring_peaks(rss, idx)
        # get activity at peak
        else:
            idxpeak = idx

        # get co-fluctuation at peaks
        etspeaks = connectivity_utils.threshold_ets_matrix(
            ets.copy(), thr=0, selected_idxs=idxpeak
        )
        peaks_rss = rss[etspeaks.nonzero()[0]]
        rss_th = np.min(peaks_rss[np.nonzero(peaks_rss)])
    # Make selection of points with edge time-series matrix
    elif "ets" in peak_detection:
        LGR.info("Selecting points with edge time-series matrix...")
        if peak_detection == "ets":
            LGR.info("Reading AUC of surrogates to perform the thresholding step...")
            hist_sum = connectivity_utils.sum_histograms(
                surrogate_events,
            )
            thr = connectivity_utils.calculate_hist_threshold(
                hist_sum, surrogate_events[0][3][:-1], percentile=95
            )
        elif peak_detection == "ets_time":
            # Initialize array for threshold
            thr = np.zeros(t)
            # initialize array for surrogate ets at each time point
            hist_time_surrogates = np.zeros([nsur, t, nbins])
            for sur_idx in range(nsur):
                hist_time_surrogates[sur_idx, :, :] = surrogate_events[sur_idx][2]

            for time_idx in range(t):
                # get first column of all sur_ets into a matrix
                # calculate histogram of all surrogate ets at time point,
                # this is still done without sparse matrix
                hist = np.sum(hist_time_surrogates[:, time_idx, :], axis=0)

                # calculate threshold for time point
                thr[time_idx] = connectivity_utils.calculate_hist_threshold(
                    hist, surrogate_events[0][3][:-1], percentile=95
                )
        # Apply threshold on edge time-series matrix
        etspeaks = connectivity_utils.threshold_ets_matrix(ets.copy(), thr)
        idxpeak = etspeaks.nonzero()[0]
        rss = np.array(np.sqrt(ets.power(2).sum(axis=1)[:, 0].flatten())).flatten()
    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.mean(etspeaks, 0)

    return ets, rss, rssr, idxpeak, etspeaks, mu, u, v, rss_th


def ev_workflow(
    data_file,
    auc_file,
    atlas,
    surr_dir,
    out_dir,
    matrix,
    te=[0],
    nsurrogates=100,
    dvars=None,
    enorm=None,
    afni_text=None,
    history_str="",
    peak_detection="rss",
    jobs=1,
):
    """
    Main function to perform event detection and plot results.
    """
    #  If te is None, make it a list with 0
    if te is None and len(data_file) == 1:
        te = [0]
    # Perform event detection on AUC
    LGR.info("Performing event-detection on AUC...")
    (ets_auc, rss_auc, _, idxpeak_auc, ets_auc_denoised, _, _, _, rss_thr) = event_detection(
        auc_file,
        atlas,
        join(surr_dir, "surrogate_AUC_"),
        nsur=nsurrogates,
        peak_detection=peak_detection,
        te=[0],
        jobs=jobs,
    )

    LGR.info("Plotting AUC, and AUC-denoised ETS matrices...")
    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(
        ets_auc.toarray(),
        out_dir,
        rss_auc,
        "_AUC_original_" + peak_detection,
        dvars,
        enorm,
        idxpeak_auc,
    )
    plot_ets_matrix(
        ets_auc_denoised.toarray(),
        out_dir,
        rss_auc,
        "_AUC_denoised_" + peak_detection,
        dvars,
        enorm,
        idxpeak_auc,
        vmax=0.02,
        thr=rss_thr,
    )

    # Save RSS time-series as text file for easier visualization on AFNI
    if afni_text is not None:
        np.savetxt(join(out_dir, afni_text) + "_peaks.txt", idxpeak_auc)
        rss_out = np.zeros(rss_auc.shape)
        if peak_detection == "rss":
            rss_out[idxpeak_auc] = rss_auc[idxpeak_auc]
            np.savetxt(join(out_dir, afni_text) + "_rss.txt", rss_auc)
            np.savetxt(join(out_dir, afni_text) + "_rss_th.txt", rss_out)
            timepoints = np.zeros(rss_auc.shape)
            timepoints[idxpeak_auc] = 1
            np.savetxt(join(out_dir, afni_text) + "_timepoints.1D", timepoints)
        if peak_detection == "ets":
            np.savetxt(join(out_dir, afni_text) + "_rss.txt", rss_auc)

    # if txt extension
    if ".txt" in matrix:
        LGR.info("Saving as .txt file...")
        np.savetxt(matrix, ets_auc_denoised.toarray())
    elif ".npz" in matrix:
        LGR.info("Saving as .npz")
        save_npz(matrix, ets_auc_denoised)
    else:
        LGR.info("No extension indicated in matrix, saving as .npz")
        save_npz(matrix, ets_auc_denoised)

    return ets_auc_denoised
