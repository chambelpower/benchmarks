import mne
from scipy.io import loadmat
from moabb.datasets.base import BaseDataset
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_continuous_onsets_and_labels(train_targets_file, test_targets_file, epoch_duration=1.4):
    # Load the targets from both train and test
    train_targets = np.loadtxt(train_targets_file, dtype=int)
    test_targets = np.loadtxt(test_targets_file, dtype=int)

    # Combine train and test targets
    all_targets = np.concatenate([train_targets, test_targets])

    # Calculate continuous onset times (ensuring length matches targets)
    onsets = np.linspace(0, (len(all_targets) - 1) * epoch_duration, len(all_targets))

    # Create labels based on target values
    labels = ["Target" if value == 1 else "NonTarget" for value in all_targets]

    # Sanity check
    if len(onsets) != len(labels):
        raise ValueError("Mismatch between onsets and labels length.")

    return onsets, labels


class BCIAUTP300V3(BaseDataset):
    """Dataset for the P300-based BCI study with ASD individuals."""

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            sessions_per_subject=7,
            events={"Target": 1, "NonTarget": 0},
            code="BCIAUTP300",
            interval=[-0.2, 1.2],  # From -200 ms to 1200 ms
            paradigm="p300",
            doi="",
        )

    def _get_single_subject_data(self, subject):
        sfreq = 250
        ch_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
        ch_types = ['eeg'] * 8
        subject_str = f'{subject:02d}' if subject < 10 else str(subject)

        # Assuming this script is in the MOABB directory or being run from it
        base_path = os.path.join(os.path.dirname(__file__), "data", f"SBJ{subject_str}")
        sessions = {}
        print('BASE PATH: ', base_path)

        for session_num in range(1, 8):
            session_str = f'S{session_num:02d}'
            session_path = os.path.join(base_path, session_str)

            # File paths for train and test data and targets
            train_data_path = os.path.join(session_path, "Train", "trainData.mat")
            train_targets_path = os.path.join(session_path, "Train", "trainTargets.txt")

            test_data_path = os.path.join(session_path, "Test", "testData.mat")
            test_targets_path = os.path.join(session_path, "Test", "testTargets.txt")

            # Load train and test data
            train_data = loadmat(train_data_path)['trainData']
            test_data = loadmat(test_data_path)['testData']

            # Reshape train and test data to [channels x (epoch * event)]
            reshaped_train_data = train_data.transpose(0, 2, 1).reshape(train_data.shape[0], -1)
            reshaped_test_data = test_data.transpose(0, 2, 1).reshape(test_data.shape[0], -1)

            # Concatenate train and test data
            concatenated_data = np.concatenate([reshaped_train_data, reshaped_test_data], axis=1)

            padding = int(sfreq * (1.2 + 0.2))  # tmax + abs(tmin)
            concatenated_data = np.pad(concatenated_data, ((0, 0), (0, padding)), mode='constant')

            # Calculate continuous onsets and labels
            onsets, labels = calculate_continuous_onsets_and_labels(train_targets_path, test_targets_path)

            # Create MNE info structure
            info = mne.create_info(ch_names, sfreq, ch_types)
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)

            # Create RawArray
            raw = mne.io.RawArray(concatenated_data, info)

            # Create annotations
            annotations = mne.Annotations(
                onset=onsets,
                duration=[0] * len(labels),
                description=labels
            )

            # Set annotations to the raw object
            raw.set_annotations(annotations)

            # Store session data
            sessions[str(session_num - 1)] = {"0": raw}

        return sessions

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        if subject not in self.subjects:
            raise ValueError("Invalid subject number.")
        return []
