import logging
import os
import subprocess
import sys
from os.path import basename
from os.path import join as opj

import numpy as np
from debiasing.debiasing_functions import debiasing_block, debiasing_spike
from nilearn.input_data import NiftiLabelsMasker
from utils import atlas_mod
from utils.hrf_generator import HRFMatrix

# AUC = "sub-002ParkMabCm_AUC_200.nii.gz"
# SUR_PREFIX = "surrogate_AUC_"
# DIR = "/bcbl/home/public/PARK_VFERRER/PFM_data"
# TEMP = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_sub-002ParkMabCm_200"
# N_SUR = 100
# DATA = "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz"
# ATLAS = "GM_200parcels_17networks_func.nii.gz"
# MASK = "sub-002ParkMabCm_T1_mask.al_epi.FUNC.nii.gz"
# DATA_DIR =
# "/bcbl/home/public/PARK_VFERRER/PREPROC/sub-002ParkMabCm/func/task-restNorm_acq-MB3_run-01"
# TR = 0.83
# BLOCK = False

LGR = logging.getLogger(__name__)


def threshold(y, thr):
    """
    Threshold input array.
    """
    y_out = y
    y_out[y_out < thr] = 0
    return y_out


def surr_debiasing(DATA, AUC, OUT, TEMP, TR, N_SUR, BLOCK, percent_th):
    """
    Main function.
    """

    SUR_PREFIX = "surrogate_AUC_"
    DIR = "/bcbl/home/public/PARK_VFERRER/PFM_data"
    # ATLAS = opj(TEMP, "atlas.nii.gz")
    output_str = "debaiasing " + AUC + "\n"
    sys.stdout.write(output_str)  # same as print
    sys.stdout.flush()
    history_str = "pySPFM debiasing."
    # Read original data.
    masker = NiftiLabelsMasker(
        labels_img=opj(TEMP, "atlas.nii.gz"),
        standardize=True,
        memory="nilearn_cache",
        strategy="mean",
    )
    data = masker.fit_transform(DATA)
    # Read AUC of original data.
    auc = masker.fit_transform(opj(DIR, AUC))

    temp_files = os.listdir(TEMP)
    surr_files = [s for s in temp_files if SUR_PREFIX in s]

    if AUC in surr_files:
        N_SUR = N_SUR - 1
        surr_files.remove(AUC)

    # Read AUC data of surrogates and save into matrix.
    sur_auc_mtx = np.zeros((auc.shape[0], auc.shape[1], N_SUR))
    for sur_idx, surr_file in enumerate(surr_files):
        LGR.info(f"Reading {surr_file}...")
        sur_auc_mtx[:, :, sur_idx] = masker.fit_transform(opj(TEMP, surr_file))

    LGR.info("All AUC data read.")

    # Calculate percentile of each ROI with data from all surrogates and apply
    # threshold to AUC of original data.
    thr = np.zeros((auc.shape[1]))

    for roi_idx in range(auc.shape[1]):
        merged_roi_auc = np.squeeze(sur_auc_mtx[:, roi_idx, :]).flatten()
        thr[roi_idx] = np.percentile(merged_roi_auc, percent_th)
        auc[:, roi_idx] = threshold(auc[:, roi_idx], thr[roi_idx])

    # Create HRF matrix
    hrf = HRFMatrix(
        TR=TR,
        TE=[0],
        nscans=data.shape[0],
        r2only=True,
        has_integrator=BLOCK,
        is_afni=True,
    )
    hrf.generate_hrf()

    # Debias data
    if BLOCK:
        deb_output = np.zeros(auc.shape)
        for roi_idx in range(auc.shape[1]):
            deb_output[:, roi_idx], _ = debiasing_block(
                auc[:, roi_idx], hrf.hrf_norm, data[:, roi_idx], is_ls=True
            )

    else:
        deb_output = debiasing_spike(hrf, data, auc)
        beta = deb_output["beta"]
        fitt = deb_output["betafitts"]
    beta_4D = masker.inverse_transform(beta)
    beta_file = opj(OUT, f"{basename(DATA[:-7])}_beta_{percent_th}.nii.gz")
    beta_4D.to_filename(beta_file)
    atlas_mod.inverse_transform(beta_file, DATA)
    subprocess.run(f"3dNotes -h {history_str} {beta_file}", shell=True)
    fitt_4D = masker.inverse_transform(fitt)
    fitt_file = opj(OUT, f"{basename(DATA[:-7])}_fitt_{percent_th}.nii.gz")
    fitt_4D.to_filename(fitt_file)
    atlas_mod.inverse_transform(fitt_file, DATA)
    subprocess.run(f"3dNotes -h {history_str} {fitt_file}", shell=True)
