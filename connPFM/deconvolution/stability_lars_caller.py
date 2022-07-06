import logging
import os

import numpy as np
from dask import compute, delayed

from connPFM.deconvolution import compute_slars

LGR = logging.getLogger(__name__)


# Check if temp directory exists
def run_stability_lars(data, hrf, temp, jobs, username, niter, maxiterfactor):
    nscans = hrf.shape[1]
    nvoxels = data.shape[1]

    # Create temp folder if it doesn't exist
    os.makedirs(temp, exist_ok=True)

    # Save data into memmap object
    data_filename = os.path.join(temp, "data.npy")
    np.save(data_filename, data)

    # Save HRF to disk
    filename_hrf = os.path.join(temp, "hrf.npy")
    np.save(filename_hrf, hrf)

    # Calculates number of TE
    nTE = int(hrf.shape[0] / nscans)

    last = 0
    LGR.info("Numer of voxels: {}".format(nvoxels))
    if jobs == 0:
        LGR.info("non paraleized option for testing")
        compute_slars.main(
            data_filename,
            filename_hrf,
            nscans,
            maxiterfactor,
            0,
            nsurrogates=niter,
            nte=nTE,
            mode=1,
            tempdir=temp,
            first=int(0),
            last=int(data.shape[1]),
            voxels_total=nvoxels,
        )
        auc_filename = temp + "/auc_" + str(0) + ".npy"
        auc = np.load(auc_filename)
    else:
        futures = []
        for job_idx in range(jobs):
            jobs_left = jobs - job_idx
            voxels_left = nvoxels - last
            voxels_job = int(np.ceil(voxels_left / jobs_left))
            if job_idx == 0:
                first = 0
                last = first + voxels_job - 1
            elif job_idx != (jobs - 1):
                first = last + 1
                last = first + voxels_job - 1
            elif job_idx == (jobs - 1):
                first = last + 1
                last = nvoxels
            LGR.info("First voxel: {}".format(first))
            LGR.info("Last voxel: {}".format(last))

            fut = delayed(compute_slars.main, pure=False)(
                data_filename,
                filename_hrf,
                nscans,
                maxiterfactor,
                job_idx,
                nsurrogates=niter,
                nte=nTE,
                mode=1,
                tempdir=temp,
                first=first,
                last=last,
                voxels_total=nvoxels,
            )
            futures.append(fut)
        compute(futures)
        for job_idx in range(jobs):
            auc_filename = temp + "/auc_" + str(job_idx) + ".npy"
            if job_idx == 0:
                auc = np.load(auc_filename)
            else:
                auc = np.hstack((auc, np.load(auc_filename)))
    return auc
