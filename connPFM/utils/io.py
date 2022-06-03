import logging
import subprocess

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.utils import atlas_mod

LGR = logging.getLogger(__name__)


def load_data(data, atlas, n_echos=1):
    """
    Load and mask data with atlas using NiftiLabelsMasker.

    Parameters
    ----------
    data : list of str
        list of datasets containing the different echos
    atlas : str
        dataset with the different ROIs to extract the timeseries
    n_echos : integer
        Number of echos

    Returns
    -------
    data_masked : Numpy matrix
        nROI x nscans Timeseries of the selected ROIs extracted from the dataset,
        in case of multiecho echoes are concatenated as nROI x (nscans x echos)
    masker : instance of NiftiMasker
        masker object to load the data
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

    Parameters
    ----------
    data : list of str
        nROI x nscans Timeseries of the selected ROIs extracted from the dataset
    output: str
        path for putput file
    masker : instance of NiftiMasker
        masker object to tramfrom the data to a 3dmatrix

    Returns
    -------
    data_masked : Numpy matrix
        nROI x nscans Timeseries of the selected ROIs extracted fromt the dataset,
        in case of multiecho echoes are concatenated as nROI x (nscans x echos)
    masker : instance of NiftiMasker
        masker object to load the data
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
