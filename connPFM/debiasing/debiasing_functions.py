import logging

import numpy as np
import scipy as sci
from sklearn.linear_model import RidgeCV

LGR = logging.getLogger(__name__)


def group_hrf(hrf, non_zero_idxs, group_dist=3):

    temp = np.zeros(hrf.shape[0])
    hrf_out = np.zeros(hrf.shape)
    non_zeros_flipped = np.flip(non_zero_idxs)
    new_idxs = []

    for iter_idx, nonzero_idx in enumerate(non_zeros_flipped):
        if (
            iter_idx != len(non_zeros_flipped) - 1
            and abs(nonzero_idx - non_zeros_flipped[iter_idx + 1]) <= group_dist
        ):
            temp += hrf[:, nonzero_idx]
        else:
            temp += hrf[:, nonzero_idx]
            hrf_out[:, nonzero_idx] = temp
            temp = np.zeros(hrf.shape[0])
            new_idxs.append(nonzero_idx)

    new_idxs = np.flip(new_idxs)
    hrf_out = hrf_out[:, new_idxs]

    return hrf_out, new_idxs


def group_betas(betas, non_zero_idxs, group_dist=3):

    for i in range(len(non_zero_idxs)):
        if i > 0 and (non_zero_idxs[i] - non_zero_idxs[i - 1] <= group_dist):
            betas[non_zero_idxs[i]] = betas[non_zero_idxs[i - 1]]

    return betas


# Performs the debiasing step on an AUC timeseries obtained considering the integrator model
def debiasing_block(auc, hrf, y, is_ls):

    # Find indexes of nonzero coefficients
    nonzero_idxs = np.where(auc != 0)[0]
    n_nonzero = len(nonzero_idxs)  # Number of nonzero coefficients

    # Initiates beta
    beta = np.zeros((y.shape))
    S = 0

    if n_nonzero != 0:
        # Initiates matrix S and array of labels
        S = np.zeros((y.shape[0], n_nonzero + 1))
        labels = np.zeros((y.shape[0]))

        # Gives values to S design matrix based on nonzeros in AUC
        # It also stores the labels of the changes in the design matrix
        # to later generate the debiased timeseries with the obtained betas
        for idx in range(n_nonzero + 1):
            if idx == 0:
                S[0 : nonzero_idxs[idx], idx] = 1
                labels[0 : nonzero_idxs[idx]] = idx
            elif idx == n_nonzero:
                S[nonzero_idxs[idx - 1] :, idx] = 1
                labels[nonzero_idxs[idx - 1] :] = idx
            else:
                S[nonzero_idxs[idx - 1] : nonzero_idxs[idx], idx] = 1
                labels[nonzero_idxs[idx - 1] : nonzero_idxs[idx]] = idx

        # Performs the least squares to obtain the beta amplitudes
        if is_ls:
            beta_amplitudes, _, _, _ = np.linalg.lstsq(
                np.dot(hrf, S), y, rcond=None
            )  # b-ax --> returns x
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]).fit(np.dot(hrf, S), y)
            beta_amplitudes = clf.coef_

        # Positions beta amplitudes in the entire timeseries
        for amp_change in range(n_nonzero):
            beta[labels == amp_change] = beta_amplitudes[amp_change]

    return (beta, S)


def debiasing_spike(x, y, beta, nlambdas=20, groups=False, group_dist=3):

    beta_out = np.zeros((x.hrf_norm.shape[1], beta.shape[1]))
    fitts_out = np.zeros(y.shape)

    x = x.hrf_norm.copy()

    index_voxels = np.unique(np.where(abs(beta) > 10 * np.finfo(float).eps)[1])

    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(abs(beta[:, index_voxels[voxidx]]) > 10 * np.finfo(float).eps)[
            0
        ]
        beta2save = np.zeros((beta_out.shape[0], 1))

        if groups:
            X_events, index_events_opt_group = group_hrf(x, index_events_opt, group_dist)
            X_fit_events = X_events.copy()

            coef_LSfitdebias, _, _, _ = sci.linalg.lstsq(
                X_events, y[:, index_voxels[voxidx]], cond=None
            )
            beta2save[index_events_opt_group, 0] = coef_LSfitdebias
            fitts_out[:, index_voxels[voxidx]] = np.dot(X_fit_events, coef_LSfitdebias)
            beta_out[:, index_voxels[voxidx]] = group_betas(
                beta2save.reshape(len(beta2save)), index_events_opt, group_dist
            )
        else:
            X_events = x[:, index_events_opt]

            coef_LSfitdebias, _, _, _ = sci.linalg.lstsq(
                X_events, y[:, index_voxels[voxidx]], cond=None
            )
            beta2save[index_events_opt, 0] = coef_LSfitdebias
            fitts_out[:, index_voxels[voxidx]] = np.dot(X_events, coef_LSfitdebias)
            beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))

        LGR.info(f"Voxel {voxidx + 1} of {len(index_voxels)} debiased...")

    # Output dictionary
    debiasing_out = {"beta": beta_out, "betafitts": fitts_out}

    return debiasing_out
