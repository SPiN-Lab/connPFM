import logging
import os
from shutil import which

from connPFM.deconvolution.stability_lars_caller import run_stability_lars
from connPFM.tests.conftest import fetch_file
from connPFM.utils import hrf_generator, surrogate_generator
from connPFM.utils.io import load_data, save_img

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
    """
    Compute PFM for the timeseries of the dataset ROIs

    Parameters
    ----------
    data : list of str
        list of datasets containing the different echos
    atlas : str
        dataset with the different ROIs to extract the timeseries
    output: str
        path for putput file
    tr: integer
        repetition time of the dataset
    username: str
        name of the user in case of launching jobs in a cluster
    te: list of integer
        echo times for the different echoes if data is multiecho
    dir: str
        temporal directory name
    block: boolean
        if True, use the block design for the hrf
    jobs: integer
        number of jobs to launch in a cluster,
        if 0 then use serial execution (recomended only for testing)
    nsurrogates: integer
        number of surrogate datasets to generate
    nstability: integer
        number of stability iterations
    percentile: float
        percentile to use for the stability threshold
    maxiterfactor: float
        factor to use for the maximum number of iterations
    hrf_shape: str
        shape of the HRF to use
    hrf_path: str
        path to the user hrf file to use
    history_str: str
        history string to add to the output file
    """
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
    if which("singularity") is not None:
        fetch_file("n7tzh", os.path.dirname(os.path.realpath(__file__)), "connpfm_slim.simg")
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
