"""Main workflow to perform event detection."""
import logging
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

from connPFM.connectivity import connectivity_utils
from connPFM.connectivity.plotting import plot_ets_matrix

LGR = logging.getLogger(__name__)


def event_detection(
    data_file,
    atlas,
    surrprefix="",
    sursufix="",
    nsur=100,
    segments=True,
    peak_detection="rss",
):
    """Perform event detection on given data."""
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        strategy="mean",
    )

    data = masker.fit_transform(data_file)
    # load and zscore time series
    # AUC does not get z-scored
    if "AUC" in surrprefix:
        z_ts = data
    else:
        z_ts = np.nan_to_num(zscore(data, ddof=1))
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # calculate ets
    ets, u, v = connectivity_utils.calculate_ets(z_ts, n)

    # Initialize thresholded edge time-series matrix with zeros
    etspeaks = np.zeros(ets.shape)

    # Initialize array of indices of detected peaks
    idxpeak = np.zeros(t)

    # Initialize RSS with zeros
    rss = np.zeros(t)

    # Initialize array for null RSS
    rssr = np.zeros([t, nsur])

    # calculate ets and rss of surrogate data
    surrogate_events = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(connectivity_utils.rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand)
        for irand in range(nsur)
    )

    hist_ranges = np.zeros((2, nsur))
    for irand in range(nsur):
        hist_ranges[0, irand] = surrogate_events[irand][2]
        hist_ranges[1, irand] = surrogate_events[irand][3]

    hist_min = np.min(hist_ranges, axis=1)[0]
    hist_max = np.max(hist_ranges, axis=1)[1]

    # Make selection of points with RSS
    if "rss" in peak_detection:
        LGR.info("Selecting points with RSS...")
        # calculate rss
        rss = np.sqrt(np.sum(np.square(ets), axis=1))

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

        # find frames that pass statistical testz_ts
        idx = np.argwhere(p_value < pcrit)[:, 0]
        if segments:
            idxpeak = connectivity_utils.remove_neighboring_peaks(rss, idx)
        # get activity at peak
        else:
            idxpeak = idx

        # get co-fluctuation at peaks
        etspeaks = connectivity_utils.threshold_ets_matrix(ets, thr=0, selected_idxs=idxpeak)

    # Make selection of points with edge time-series matrix
    elif "ets" in peak_detection:
        LGR.info("Selecting points with edge time-series matrix...")
        if peak_detection == "ets":
            LGR.info("Reading AUC of surrogates to perform the thresholding step...")
            all_hist = True
            if n > 100000:
                all_hist = False
            thr = connectivity_utils.surrogates_histogram(
                surrprefix,
                sursufix,
                masker,
                hist_range=(hist_min, hist_max),
                numrand=nsur,
                all_hist=all_hist,
            )
        elif peak_detection == "ets_time":
            # Initialize array for threshold
            thr = np.zeros(t)

            # initialize array for surrogate ets at each time point
            sur_ets_at_time = np.zeros((nsur, surrogate_events[0][1].shape[1]))

            for time_idx in range(t):
                # get first column of all sur_ets into a matrix
                for sur_idx in range(nsur):
                    sur_ets_at_time[sur_idx, :] = surrogate_events[sur_idx][1][time_idx, :]

                # calculate histogram of all surrogate ets at time point
                hist, bins = np.histogram(
                    sur_ets_at_time.flatten(), bins=500, range=(hist_min, hist_max)
                )

                # calculate threshold for time point
                thr[time_idx] = connectivity_utils.calculate_hist_threshold(
                    hist, bins, percentile=95
                )

        # Apply threshold on edge time-series matrix
        etspeaks = connectivity_utils.threshold_ets_matrix(ets, thr)
        idxpeak = np.where(etspeaks != 0)[0]
        rss = np.sqrt(np.sum(np.square(etspeaks), axis=1))
    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.nanmean(etspeaks, 0)

    return ets, rss, rssr, idxpeak, etspeaks, mu, u, v


def ev_workflow(
    data_file,
    auc_file,
    atlas,
    surr_dir,
    out_dir,
    matrix,
    nsurrogates=100,
    dvars=None,
    enorm=None,
    afni_text=None,
    history_str="",
    peak_detection="rss",
):
    """
    Main function to perform event detection and plot results.
    """
    # Paths to files
    # Perform event detection on ORIGINAL data
    LGR.info("Performing event-detection on original data...")
    (
        _,
        rss_orig,
        _,
        idxpeak_orig,
        ets_orig_denoised,
        _,
        _,
        _,
    ) = event_detection(data_file, atlas, join(surr_dir, "surrogate_"), nsur=nsurrogates)

    # Perform event detection on AUC
    LGR.info("Performing event-detection on AUC...")
    (ets_auc, rss_auc, _, idxpeak_auc, ets_auc_denoised, _, _, _,) = event_detection(
        auc_file,
        atlas,
        join(surr_dir, "surrogate_AUC_"),
        nsur=nsurrogates,
        peak_detection=peak_detection,
    )

    LGR.info("Plotting original, AUC, and AUC-denoised ETS matrices...")
    plot_ets_matrix(
        ets_orig_denoised,
        out_dir,
        rss_orig,
        "_original_" + peak_detection,
        dvars,
        enorm,
        idxpeak_orig,
    )
    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(
        ets_auc, out_dir, rss_auc, "_AUC_original_" + peak_detection, dvars, enorm, idxpeak_auc
    )
    plot_ets_matrix(
        ets_auc_denoised,
        out_dir,
        rss_auc,
        "_AUC_denoised_" + peak_detection,
        dvars,
        enorm,
        idxpeak_auc,
        vmax=0.02,
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

    np.savetxt(matrix, ets_auc_denoised)

    return ets_auc_denoised
