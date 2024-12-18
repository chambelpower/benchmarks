import numpy as np
import torch
from scipy.signal import welch
from scipy.fft import fft, ifft

def frequency_shift(eeg_data, fs=128, delta_f_max=2, probability=0.5):
    """
    Applies frequency shift augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - fs: Sampling frequency
    - delta_f_max: Maximum frequency shift magnitude
    - probability: Probability of applying the frequency shift to each window

    Returns:
    - shifted_eeg_data: Frequency-shifted EEG data as a PyTorch tensor
    """
    eeg_data_np = eeg_data.numpy()  # Convert to numpy for processing
    shifted_eeg_data_np = np.zeros_like(eeg_data_np)

    for sample_idx, sample in enumerate(eeg_data_np):
        for channel_idx, channel in enumerate(sample.T):  # Transpose to iterate over channels
            if np.random.rand() < probability:
                f, Pxx = welch(channel, fs=fs)
                delta_f = np.random.uniform(-delta_f_max, delta_f_max)

                # Shift the frequency
                shifted_Pxx = np.roll(Pxx, int(delta_f * len(f) / (fs / 2)))

                # Inverse FFT to get the time-domain signal back
                shifted_channel = ifft(np.sqrt(shifted_Pxx)).real
                shifted_channel = np.interp(np.arange(len(channel)), np.arange(len(shifted_channel)), shifted_channel)
                shifted_eeg_data_np[sample_idx, :, channel_idx] = shifted_channel
            else:
                shifted_eeg_data_np[sample_idx, :, channel_idx] = channel

    shifted_eeg_data = torch.from_numpy(shifted_eeg_data_np).float()  # Convert back to PyTorch tensor

    return shifted_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
shifted_train_data = frequency_shift(train_data)

print(train_data.shape)
print(shifted_train_data.shape)
