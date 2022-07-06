#!/usr/bin/python
import logging
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from connPFM.deconvolution.stability_lars import StabilityLars

LGR = logging.getLogger(__name__)


def main(
    data_filename,
    hrf_file,
    nscans,
    maxiterfactor,
    n_job,
    voxel=1,
    nsurrogates=100,
    nte=1,
    mode=1,
    tempdir=None,
    first=None,
    last=None,
    voxels_total=1,
):

    LGR.info("Data file is ", data_filename)
    LGR.info("HRF file is ", hrf_file)
    LGR.info("Voxel is: ", voxel)
    LGR.info("nscans is: ", nscans)
    LGR.info("N surrogates: ", maxiterfactor)
    LGR.info("N TE: ", nte)
    LGR.info("Mode: ", mode)
    LGR.info("Dir: ", tempdir)

    sl = StabilityLars()

    sl.nsurrogates = nsurrogates
    sl.nTE = nte
    sl.mode = mode

    if first is None:
        nvoxels = 1
    else:
        nvoxels = last - first + 1
        voxel = first

    if last == voxels_total:
        nvoxels = nvoxels - 1

    y = np.load(data_filename)
    auc = np.zeros((nscans, nvoxels))
    sl.key = first
    sl.maxiterfactor = maxiterfactor

    LGR.info("Job number: {}".format(n_job))
    LGR.info("First voxel: {}".format(first))
    LGR.info("Last voxel: {}".format(last))
    LGR.info("Number of voxels: {}".format(nvoxels))

    hrf = np.load(hrf_file)
    for vox_idx in range(nvoxels):
        sl.stability_lars(hrf, np.expand_dims(y[:, voxel + vox_idx], axis=-1))
        auc[:, vox_idx] = np.squeeze(sl.auc)
        LGR.info(
            "AUC of voxel {}/{} calculated and stored...".format(str(vox_idx + 1), str(nvoxels))
        )

    filename = os.path.join(tempdir, "auc_" + str(n_job) + ".npy")
    np.save(filename, auc)


if __name__ == "__main__":
    main(sys.argv[1:])
