import logging
import os

from connPFM.deconvolution.stability_lars_caller import run_stability_lars
from connPFM.utils import hrf_generator, surrogate_generator
from connPFM.utils.io import dask_scheduler, load_data, save_img

LGR = logging.getLogger(__name__)


def roiPFM(
    data,
    atlas,
    output,
    tr,
    username,
    te=None,
    dir="temp",
    block=False,
    jobs=50,
    nsurrogates=50,
    nstability=50,
    percentile=95,
    maxiterfactor=0.3,
    hrf_shape="SPMG1",
    hrf_path=None,
    history_str="",
):

    if te is None and len(data) == 1:
        te = [0]
    elif len(te) > 1:
        # If all values in TE list are higher than 1, divide them by 1000.
        # Only relevant for multi-echo data.
        if all(te_val > 1 for te_val in te):
            te = [te_val / 1000 for te_val in te]

    os.makedirs(dir, exist_ok=True)

    LGR.info("Masking data...")
    data_masked, masker = load_data(data, atlas, n_echos=len(te))
    LGR.info("Data masked.")

    LGR.info("Generating HRF...")
    # Generates HRF matrix
    hrf_matrix = hrf_generator.HRFMatrix(
        TR=tr,
        TE=te,
        nscans=int(data_masked.shape[0] / len(te)),
        r2only=True,
        block=block,
        is_afni=True,
        lop_hrf=hrf_shape,
        path=hrf_path,
    )
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.hrf_norm
    LGR.info("HRF generated.")
    LGR.info("Running stability selection on original data...")

    _, cluster = dask_scheduler(jobs)
    auc = run_stability_lars(
        data=data_masked,
        hrf=hrf,
        temp=dir,
        jobs=jobs,
        username=username,
        niter=nstability,
        maxiterfactor=maxiterfactor,
    )
    LGR.info("Stability selection on original data finished.")

    LGR.info("Saving AUC results of original data...")
    save_img(auc, output, masker, history_str)
    LGR.info("AUC results on original data saved.")

    if nsurrogates:
        LGR.info(f"Performing PFM on {nsurrogates} surrogates...")
        LGR.info("Make yourself a cup of coffee while it runs :)")
        for n_sur in range(nsurrogates):
            # Generate surrogate
            surrogate_name = os.path.join(dir, f"surrogate_{n_sur}.nii.gz")
            surrogate_masked = surrogate_generator.generate_surrogate(
                data_masked, masker, output=surrogate_name
            )
            # Calculate AUC
            auc = run_stability_lars(
                data=surrogate_masked,
                hrf=hrf,
                temp=dir,
                jobs=jobs,
                username=username,
                niter=nstability,
                maxiterfactor=maxiterfactor,
            )

            # Save surrogate AUC
            surrogate_out = os.path.join(dir, f"surrogate_AUC_{n_sur}.nii.gz")
            save_img(auc, surrogate_out, masker, history_str)
            LGR.info(f"{n_sur}/{nsurrogates -1 }")

        LGR.info(f"PFM on {nsurrogates} surrogates finished.")

    LGR.info("PFM finished.")
