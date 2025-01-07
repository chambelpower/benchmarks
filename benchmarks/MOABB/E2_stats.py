#https://we.tl/t-Z8rYPmSkre

import os
import numpy as np
from scipy.io import loadmat
from models.EEGNet import EEGNet
import torch
from utils.dataio_iterators import crop_signals
import csv
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.signal import resample
from mne.filter import filter_data
from mne.filter import resample
import subprocess
from torch.utils.data import TensorDataset, DataLoader
import pickle

def model_single_predict(model, eeg_sample):
    """
    Predict probabilities for a single EEG sample.

    Parameters:
        model: The trained model to use for predictions.
        eeg_sample (ndarray): Single EEG sample with shape [channels x timepoints].

    Returns:
        prob_p300 (float): Predicted probability of the P300 class.
        prob_non_p300 (float): Predicted probability of the non-P300 class.
    """
    # Expand dimensions to match the required input shape [1, C, T, 1]
    eeg_tensor = torch.tensor(eeg_sample, dtype=torch.float32)

    eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(3)
    #print('shape: ', eeg_tensor.shape)

    # Forward pass through the model
    output = model(eeg_tensor)

    #print('OUTPUT: ', output)

    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    prob_p300 = probs[0, 1].item()  # Probability of P300
    prob_non_p300 = probs[0, 0].item()  # Probability of Non-P300

    return prob_p300, prob_non_p300

def predict(model, eeg_data):
    """
    Predict P300 labels for pre-split EEG samples.

    Parameters:
        model: The trained model to use for predictions.
        eeg_data (ndarray): EEG data with shape [samples x timepoints x channels].

    Returns:
        predictions (list): Predicted labels (0 or 1) for each sample.
    """
    bin_predictions = []
    p300_prob_array = []

    for eeg_sample in eeg_data:

        #print('~SHAPE: ', eeg_sample.shape)
        # Predict probabilities for the single sample
        prob_p300, prob_non_p300 = model_single_predict(model, eeg_sample)

        p300_prob_array.append(prob_p300)
        
        # Determine the predicted label
        if prob_p300 > prob_non_p300:
            bin_predictions.append(1)
        elif prob_p300 < prob_non_p300:
            bin_predictions.append(0)
        else:
            bin_predictions.append(0)  # Default to non-P300 in case of a tie

    return bin_predictions, p300_prob_array



def evaluate(bin_predictions, p300_prob_array, labels):
    """Evaluates the model's predictions."""
    # Binary accuracy
    bin_accuracy = np.mean(bin_predictions == labels)

    # Final accuracy based on blocks of 8
    n_blocks = len(labels) // 8  # Number of blocks
    correct = 0

    for i in range(n_blocks):
        start_idx = i * 8
        end_idx = start_idx + 8

        # Extract the block of P300 probabilities and corresponding labels
        block_probs = p300_prob_array[start_idx:end_idx]
        block_labels = labels[start_idx:end_idx]

        #print(block_probs)
        #print(block_labels)

        # Find the index of the highest probability in the block
        max_prob_idx = np.argmax(block_probs)

        # Check if the label at this index is 1
        if block_labels[max_prob_idx] == 1:
            correct += 1

    # Calculate final accuracy
    final_accuracy = correct / n_blocks

    return bin_accuracy, final_accuracy


def load_model(model_path, experiment_number):

    model_params = {
        'input_shape': [None, 125, 8, None],
        'cnn_temporal_kernels': 58,
        'cnn_temporal_kernelsize': [42, 1],
        'cnn_spatial_depth_multiplier': 3,
        'cnn_spatial_max_norm': 1,
        'cnn_spatial_pool': [4, 1],
        'cnn_septemporal_depth_multiplier': 1,
        'cnn_septemporal_point_kernels': 219,
        'cnn_septemporal_kernelsize': [17, 1],
        'cnn_septemporal_pool': [4, 1],
        'cnn_pool_type': 'avg',
        'activation_type': 'elu',
        'dense_max_norm': 0.25,
        'dropout': 0.3903,
        'dense_n_neurons': 2  
    }
   
    model = EEGNet(**model_params)

    checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)

    model.eval()

    return model

def load_data_v2(hparams, output_folder, target_subject_idx, target_session_idx):
    # Define the command and its arguments
    command = [
        "python",
        "train_v2.py",
        hparams,
        "--data_folder=eeg_data",
        "--cached_data_folder=eeg_pickled_data",
        "--output_folder=" + output_folder,
        "--target_subject_idx=" + target_subject_idx,
        "--target_session_idx=" + target_session_idx,
        "--data_iterator_name=leave-one-session-out",
        "--device=cpu"
    ]

    # Execute the command
    try:
        result = subprocess.run(command, check=True)


    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
        return

    with open("test_dataset.pkl", "rb") as f:
        test_data = pickle.load(f)

    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    print("Test dataset loaded successfully")

    print('X: ', x_test.shape)
    #print(x_test)
    print('Y: ', y_test.shape)
    #print(y_test)

    return x_test, y_test


def evaluate_model(subject_path, model_path, experiment_number, subject, session):

    concatenated_data, all_targets = load_data_v2('hparams/P300/BCIAUTP300/EEGNet' + experiment_number +'.yaml', 'results/P300/BCIAUTP300/' + experiment_number + '/', str(int(subject)), session)    

    model = load_model(model_path, experiment_number)

    print('Subject from: ', subject_path, ' LOADED')

    print('Model from: ', model_path, ' LOADED')

    # Predict and evaluate
    
    bin_predictions, p300_prob_array = predict(model, concatenated_data)
    bin_accuracy, final_accuracy = evaluate(bin_predictions, p300_prob_array, all_targets)
    
    print(f"Binary Model Accuracy: {bin_accuracy * 100:.2f}%")
    print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")

    return bin_accuracy, final_accuracy


# Helper function to get the most recent folder in a directory
def get_most_recent_folder_path(parent_path):
    try:
        folders = [f for f in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, f))]
        if not folders:
            raise ValueError(f"No folders found in {parent_path}")
        
        # Sort folders by last modification time
        folders.sort(key=lambda f: os.path.getmtime(os.path.join(parent_path, f)), reverse=True)
        
        # Return the most recent folder
        return os.path.join(parent_path, folders[0])
    except Exception as e:
        raise ValueError(f"Error accessing folder in {parent_path}: {e}")

if __name__ == "__main__":

    # Specific experiment get stats about
    experiment = 'E2'

    # Subjects and runs (update with your specific structure)
    runs = [f"run{i}" for i in range(1, 11)]  
    subjects = [f"sub-{str(i).zfill(3)}" for i in range(1, 16)] 
    sessions = [str(j) for j in range(7)] 

    bin_accuracies = {}
    final_accuracy = {}
 
    # Iterate through each run, subject, and session
    for run in runs:
        for subject in subjects:
            for session in sessions:
                #try:
                path1 = os.path.join(os.path.dirname(__file__),'results', 'P300', 'BCIAUTP300' , experiment, run)
                
                seed_folder = get_most_recent_folder_path(path1)

                path2 = os.path.join(seed_folder, 'leave-one-session-out', subject, session, 'save')
                
                timestamp_folder = get_most_recent_folder_path(path2)

                model_path = os.path.join(timestamp_folder, 'model.ckpt')

                path3 = os.path.join(os.path.dirname(__file__), 'data', 'SBJ' + str(subject[-2]) + str(subject[-1]), 'S0' + str(int(session) + 1))

                acc1, acc2 = evaluate_model(path3, model_path, experiment[-2:], str(subject[-2]) + str(subject[-1]), session)

                model_info = run + subject + "sess" + session

                bin_accuracies[model_info] = acc1
                final_accuracy[model_info] = acc2
                    
                #except Exception as e:
                #    print(f"Error processing {run}, {subject}, session {session}: {e}")

    # Specify the TXT file path
    file_path = experiment + "_binary_accuracies.txt"

    # Write to the TXT file
    with open(file_path, "w") as file:
        for accuracy in bin_accuracies.values():
            # Write each accuracy on a new line
            file.write(f"{accuracy* 100:.2f}\n")

    print(f"Binary accuracies saved to {file_path}")

    # Specify the TXT file path
    file_path = experiment + "_final_accuracies.txt"

    # Write to the TXT file
    with open(file_path, "w") as file:
        for accuracy in final_accuracy.values():
            # Write each accuracy on a new line
            file.write(f"{accuracy* 100:.2f}\n")

    print(f"Final accuracies saved to {file_path}")
