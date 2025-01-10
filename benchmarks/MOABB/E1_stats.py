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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

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
    Predict P300 targets for pre-split EEG samples.

    Parameters:
        model: The trained model to use for predictions.
        eeg_data (ndarray): EEG data with shape [samples x timepoints x channels].

    Returns:
        predictions (list): Predicted targets (0 or 1) for each sample.
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

def calculate_metrics(y_true, y_pred, y_prob=None):

    if len(set(y_true)) == 2:  # Binary classification
        average = 'binary'
    else:  # Multiclass classification
        average = 'macro'

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average) 
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    mcc = matthews_corrcoef(y_true, y_pred)

    if average == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
    else:
        specificity = None
    
    if y_prob is not None:
        if average == 'binary':
            auc_roc = roc_auc_score(y_true, y_prob)
        else:
            auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovo', average=average)
    else:
        auc_roc = None
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if specificity is not None:
        print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    else:
        print("AUC-ROC: Not calculated (requires predicted probabilities).")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "mcc": mcc,
        "auc_roc": auc_roc
    }

def evaluate(bin_predictions, p300_prob_array, targets, events, labels, runs_per_block_test):
    """Evaluates the model's predictions."""

    bin_stats = calculate_metrics(targets, bin_predictions, p300_prob_array)

    final_pred = []
    final_pred_prob = []
    
    i = 0
    while i < len(targets):
        # Initialize target_array as a NumPy array of zeros
        target_array = np.zeros(8)
        
        if i < 1600:  # Meaning it's training
            n_blocks = 10
        else:
            n_blocks = runs_per_block_test

        for j in range(n_blocks):
            for k in range(8):
                event_idx = events[i]
                target_array[event_idx - 1] += p300_prob_array[i]
                i += 1

        # Normalize the target_array to make it sum to 1
        total = np.sum(target_array)
        if total > 0:
            target_array /= total  # Normalize so that the sum of the array equals 1

        # Find the index of the maximum value in target_array
        m = np.max(target_array)

        # Append the class with the highest probability
        final_pred.append(np.argmax(target_array) + 1)
        # Normalize the target_array by dividing by 8
        final_pred_prob.append(target_array)

    final_stats = calculate_metrics(labels, final_pred, final_pred_prob)

    return bin_stats, final_stats



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

def load_data_v2(hparams, output_folder, target_subject_idx, target_session_idx, subject_path):
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

    #print('X: ', x_test.shape)
    #print(x_test)
    #print('Y: ', y_test.shape)
    #print(y_test)


    train_events = np.loadtxt(os.path.join(subject_path, 'Train', 'trainEvents.txt'), dtype=int)
    test_events = np.loadtxt(os.path.join(subject_path, 'Test', 'testEvents.txt'), dtype=int)
    total_events = np.concatenate((train_events, test_events))

    train_labels = np.loadtxt(os.path.join(subject_path, 'Train', 'trainLabels.txt'), dtype=int)
    test_labels = np.loadtxt(os.path.join(subject_path, 'Test', 'testLabels.txt'), dtype=int)
    total_labels = np.concatenate((train_labels, test_labels))
    
    runs_per_block_test = np.loadtxt(os.path.join(subject_path, 'Test', 'runs_per_block.txt'), dtype=int)

    return x_test, y_test, total_events, total_labels, runs_per_block_test


def evaluate_model(subject_path, model_path, experiment_number, subject, session):

    concatenated_data, all_targets, events, labels, runs_per_block_test = load_data_v2('hparams/P300/BCIAUTP300/EEGNet' + experiment_number +'.yaml', 'results/P300/BCIAUTP300/' + experiment_number + '/', subject, session, subject_path)    

    model = load_model(model_path, experiment_number)

    print('Subject from: ', subject_path, ' LOADED')

    print('Model from: ', model_path, ' LOADED')

    # Predict and evaluate
    
    bin_predictions, p300_prob_array = predict(model, concatenated_data)
    bin_stats, final_stats = evaluate(bin_predictions, p300_prob_array, all_targets, events, labels, runs_per_block_test)


    return bin_stats, final_stats


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

def write_all_metrics_to_file(all_metrics, file_path):
    """
    Write the metrics for all models to a file.

    Parameters:
    - all_metrics: list of dict, Each dict contains metrics for one model.
    - file_path: str, Path to the output file.
    """
    with open(file_path, "w") as file:
        # Write header
        headers = ["accuracy", "precision", "recall", "f1_score", "specificity", "mcc", "auc_roc"]
        file.write("\t".join(headers) + "\n")
        
        # Write metrics for each model
        for metrics in all_metrics:
            if isinstance(metrics, dict):  # Ensure that metrics is a dictionary
                line = "\t".join(
                    f"{metrics.get(metric, 'N/A') * 100:.2f}" if metrics.get(metric) is not None else "N/A"
                    for metric in headers
                )
                file.write(line + "\n")
            else:
                file.write("Invalid metrics data\n")


if __name__ == "__main__":

    # Specific experiment get stats about
    experiment = 'E1'

    # Subjects and runs (update with your specific structure)
    runs = [f"run{i}" for i in range(1, 2)]  
    subjects = [f"sub-{str(i).zfill(3)}" for i in range(1, 2)] 
    sessions = [str(j) for j in range(2)] 

    bin_stats = {}
    final_stats = {}
 
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

                stats1, stats2 = evaluate_model(path3, model_path, experiment[-1:], str(int(subject[-2]) * 10 + int(subject[-1]) - 1), session)

                model_info = run + subject + "sess" + session

                bin_stats[model_info] = stats1
                final_stats[model_info] = stats2
                    
                #except Exception as e:
                #    print(f"Error processing {run}, {subject}, session {session}: {e}")

    # Specify the TXT file path
    file_path = experiment + "_binary_stats.txt"
    all_metrics = list(bin_stats.values())

    write_all_metrics_to_file(all_metrics, file_path)

    # Specify the TXT file path
    file_path = experiment + "_final_stats.txt"
    all_metrics = list(final_stats.values())

    write_all_metrics_to_file(all_metrics, file_path)

    
