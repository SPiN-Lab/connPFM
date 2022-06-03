"""Main debiasing workflow."""
import logging

import numpy as np

from connPFM.debiasing.debiasing_functions import debiasing_spike  # or debiasing_block
from connPFM.utils import io
from connPFM.utils.hrf_generator import HRFMatrix

LGR = logging.getLogger(__name__)


def debiasing(data_file, mask, te, mtx, tr, prefix, groups, groups_dist, history_str):
    """
    Perform debiasing based on denoised edge-time matrix.

    Parameters
    ----------
    data_file : str or list of str
        Path to data files
    mask : str
        Path to mask file
    te : int
        list of TEs to perform the debiasing
    mtx : ndarray
        matrix to do the debiasing
    tr : float
        repetition time
    prefix : str
        prefix for output files
    groups : bool
        If True, perform debiasing with groups hrf
    groups_dist : int
        Distance between groups
    history_str : str
        History string
    """

    if te is None and len(data_file) == 1:
        te = [0]
    elif len(te) > 1:
        # If all values in TE list are higher than 1, divide them by 1000.
        # Only relevant for multi-echo data.
        if all(te_val > 1 for te_val in te):
            te = [te_val / 1000 for te_val in te]

    LGR.info("Performing debiasing based on denoised edge-time matrix...")
    # Read data
    data, masker = io.load_data(data_file, mask, n_echos=len(te))

    # Get number of nodes
    [_, n] = data.shape

    # Get ETS indexes
    idx_u, idx_v = np.argwhere(np.triu(np.ones(n), 1)).T

    # Generate mask of significant edge-time connections
    ets_mask = np.zeros(data.shape)
    idxs = np.where(mtx != 0)
    time_idxs = idxs[0]
    edge_idxs = idxs[1]

    LGR.info("Generating mask of significant edge-time connections...")
    for idx, time_idx in enumerate(time_idxs):
        ets_mask[time_idx, idx_u[edge_idxs[idx]]] = 1
        ets_mask[time_idx, idx_v[edge_idxs[idx]]] = 1

    # Create HRF matrix
    hrf = HRFMatrix(
        TR=tr,
        TE=te,
        nscans=int(data.shape[0] / len(te)),
        r2only=True,
        is_afni=True,
    )
    hrf.generate_hrf()
    # Perform debiasing
    deb_output = debiasing_spike(hrf, data, ets_mask, groups=groups, group_dist=groups_dist)
    beta = deb_output["beta"]
    fitt = deb_output["betafitts"]

    # Transform results back to 4D
    beta_file = f"{prefix}_beta_ETS.nii.gz"
    io.save_img(beta, beta_file, masker, history_str)

    # If n_echos is 1, save betafitts as they are.
    # If n_echos is > 1, loop through all echoes and
    # save the betaffits of each echo as a separate file.
    if len(te) == 1:
        fitt_file = f"{prefix}_fitt_ETS.nii.gz"
        io.save_img(fitt, fitt_file, masker, history_str)
    else:
        for echo_idx in range(len(te)):
            # The number of scans is the shape[1] of the hrf matrix
            nscans = hrf.hrf_norm.shape[1]

            #  Get the betafitts of the current echo from the betafitts matrix
            echo_fitt = fitt[echo_idx * nscans : (echo_idx + 1) * nscans, :]

            #  Save the betafitts of the current echo
            fitt_file = f"{prefix}_fitt_ETS_echo-{echo_idx}.nii.gz"
            io.save_img(echo_fitt, fitt_file, masker, history_str)

    LGR.info("Debiasing finished and files saved.")

    return beta, fitt
