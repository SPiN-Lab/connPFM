import logging
import subprocess
from os.path import expanduser, join

import numpy as np
import yaml
from dask import config
from dask.distributed import Client
from dask_jobqueue import PBSCluster, SGECluster, SLURMCluster
from nilearn.input_data import NiftiLabelsMasker

from connPFM.utils import atlas_mod

LGR = logging.getLogger(__name__)


def load_data(data, atlas, n_echos=1):
    """
    Load and mask data with atlas using NiftiLabelsMasker.
    """
    # Initialize masker object
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False, strategy="mean")

    # If n_echos is 1 (single echo), mask and return data
    if n_echos == 1:
        # If data is a list, keep only first element
        if isinstance(data, list):
            data = data[0]
        data_masked = masker.fit_transform(data)
    else:
        # If n_echos is > 1 (multi-echo), mask each echo in data list separately and
        # concatenate the masked data.
        # If n_echos and len(data) are equal, read data.
        if n_echos == len(data):
            for echo_idx, echo in enumerate(data):
                if echo_idx == 0:
                    data_masked = masker.fit_transform(echo)
                else:
                    data_masked = np.concatenate((data_masked, masker.fit_transform(echo)), axis=0)
        #  If n_echos is different from len(data), raise error.
        else:
            raise ValueError("Please provide as many TE as input files.")

    return data_masked, masker


def save_img(data, output, masker, history_str=None):
    """
    Save data as Nifti image, and update header history.
    """
    # Transform data back to Nifti image
    data_img = masker.inverse_transform(data)

    # Save data as Nifti image
    data_img.to_filename(output)

    # Transform space of image
    atlas_mod.inverse_transform(output)

    # If history_str is not None, update header history
    if history_str is not None:
        LGR.info("Updating file history...")
        subprocess.run('3dNotes -h "' + history_str + '" ' + output, shell=True)
        LGR.info("File history updated.")


def dask_scheduler(jobs):
    """
    Checks if the user has a dask_jobqueue configuration file, and if so,
    returns the appropriate scheduler according to the file parameters
    """
    # look if default ~ .config/dask/jobqueue.yaml exists
    with open(join(expanduser("~"), ".config/dask/jobqueue.yaml"), "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    if data is None:
        LGR.warning(
            "dask configuration wasn't detected, "
            "if you are using a cluster please look at "
            "the jobqueue YAML example, modify it so it works in your cluster "
            "and add it to ~/.config/dask "
            "local configuration will be used."
            "You can find a jobqueue YAML example in the connPFM/utils/jobqueue.yaml file."
        )
        cluster = None
    else:
        config.set(distributed__comm__timeouts__tcp="90s")
        config.set(distributed__comm__timeouts__connect="90s")
        config.set(scheduler="single-threaded")
        config.set({"distributed.scheduler.allowed-failures": 50})
        config.set(admin__tick__limit="3h")
        if "sge" in data["jobqueue"]:
            cluster = SGECluster()
            cluster.scale(jobs)
        elif "pbs" in data["jobqueue"]:
            cluster = PBSCluster()
            cluster.scale(jobs)
        elif "slurm" in data["jobqueue"]:
            cluster = SLURMCluster()
            cluster.scale(jobs)
        else:
            LGR.warning(
                "dask configuration wasn't detected, "
                "if you are using a cluster please look at "
                "the jobqueue YAML example, modify it so it works in your cluster "
                "and add it to ~/.config/dask "
                "local configuration will be used."
                "You can find a jobqueue YAML example in the connPFM/utils/jobqueue.yaml file."
            )
            cluster = None
    if cluster is None:
        client = None
    else:
        client = Client(cluster)
    return client, cluster
