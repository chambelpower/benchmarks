#!/usr/bin/python
"""
This script implements raining neural networks to decode single EEG trials using various paradigms on MOABB datasets.
For a list of supported datasets and paradigms, please refer to the official documentation at http://moabb.neurotechx.com/docs/api.html.

To run training (e.g., architecture: EEGNet; dataset: BNCI2014001) for a specific subject, recording session and training strategy:
    > python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --target_subject_idx=0 --target_session_idx=0 --data_iterator_name=leave-one-session-out

Author
------
Davide Borra, 2022
Mirco Ravanelli, 2023
"""

import pickle
import os
import torch
from hyperpyyaml import load_hyperpyyaml
from torch.nn import init
import numpy as np
import logging
import sys
from utils.dataio_iterators_v2 import LeaveOneSessionOut, LeaveOneSubjectOut
from torchinfo import summary
import speechbrain as sb
import yaml

def prepare_dataset_iterators(hparams):
    """Preprocesses the dataset and partitions it into train, valid and test sets."""
    # defining data iterator to use
    print("Prepare dataset iterators...")
    if hparams["data_iterator_name"] == "leave-one-session-out":
        data_iterator = LeaveOneSessionOut(
            seed=hparams["seed"]
        )  # within-subject and cross-session
    elif hparams["data_iterator_name"] == "leave-one-subject-out":
        data_iterator = LeaveOneSubjectOut(
            seed=hparams["seed"]
        )  # cross-subject and cross-session
    else:
        raise ValueError(
            "Unknown data_iterator_name: %s" % hparams["data_iterator_name"]
        )

    data_iterator.prepare(
        data_folder=hparams["data_folder"],
        dataset=hparams["dataset"],
        cached_data_folder=hparams["cached_data_folder"],
        batch_size=hparams["batch_size"],
        valid_ratio=hparams["valid_ratio"],
        target_subject_idx=hparams["target_subject_idx"],
        target_session_idx=hparams["target_session_idx"],
        events_to_load=hparams["events_to_load"],
        original_sample_rate=hparams["original_sample_rate"],
        sample_rate=hparams["sample_rate"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        tmin=hparams["tmin"],
        tmax=hparams["tmax"],
        save_prepared_dataset=hparams["save_prepared_dataset"],
        n_steps_channel_selection=hparams["n_steps_channel_selection"],
        seed_nodes=hparams.get("seed_nodes", ["Cz"]),
    )
  


def load_hparams_and_dataset_iterators(hparams_file, run_opts, overrides):
    """Loads the hparams and datasets, injecting appropriate overrides
    for the shape of the dataset.
    """
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    prepare_dataset_iterators(hparams)
    


if __name__ == "__main__":
    argv = sys.argv[1:]
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    overrides = yaml.load(
        overrides, yaml.SafeLoader
    )  # Convert overrides to a dict
    load_hparams_and_dataset_iterators(
        hparams_file, run_opts, overrides
    )

  
