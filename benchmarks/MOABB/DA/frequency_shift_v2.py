import numpy as np
import torch
from scipy.signal import welch
from scipy.fft import fft, ifft
import torch.nn as nn

class FrequencyShift(nn.Module):
    def __init__(self, fs=128, delta_f_max=2, probability=0.5):
        """
        Initializes the FrequencyShift module.
        
        Parameters:
        - fs: Sampling frequency
        - delta_f_max: Maximum frequency shift magnitude
        - probability: Probability of applying the frequency shift to each window
        """
        super(FrequencyShift, self).__init__()
        self.fs = fs
        self.delta_f_max = delta_f_max
        self.probability = probability

    def forward(self, waveforms):
        """
        Applies frequency shift augmentation to EEG data.

        Parameters:
        - waveforms: PyTorch tensor of EEG data (shape: samples x time x channels)

        Returns:
        - shifted_eeg_data: Frequency-shifted EEG data as a PyTorch tensor
        """
        # Save the device of the input tensor
        device = waveforms.device

        # Move tensor to CPU if it is on GPU
        if waveforms.is_cuda:
            waveforms = waveforms.cpu()

        # Convert to numpy for processing
        eeg_data_np = waveforms.numpy()
        shifted_eeg_data_np = np.zeros_like(eeg_data_np)

        for sample_idx, sample in enumerate(eeg_data_np):
            for channel_idx, channel in enumerate(sample.T):  # Transpose to iterate over channels
                if np.random.rand() < self.probability:
                    # Set nperseg to a value suitable for the input signal length
                    nperseg = min(len(channel), 125)
                    
                    f, Pxx = welch(channel, fs=self.fs, nperseg=nperseg)
                    delta_f = np.random.uniform(-self.delta_f_max, self.delta_f_max)

                    # Shift the frequency
                    shifted_Pxx = np.roll(Pxx, int(delta_f * len(f) / (self.fs / 2)))

                    # Inverse FFT to get the time-domain signal back
                    shifted_channel = ifft(np.sqrt(shifted_Pxx)).real
                    shifted_channel = np.interp(np.arange(len(channel)), np.arange(len(shifted_channel)), shifted_channel)
                    shifted_eeg_data_np[sample_idx, :, channel_idx] = shifted_channel
                else:
                    shifted_eeg_data_np[sample_idx, :, channel_idx] = channel

        # Convert back to PyTorch tensor
        shifted_eeg_data = torch.from_numpy(shifted_eeg_data_np).float()

        # Move back to the original device
        shifted_eeg_data = shifted_eeg_data.to(device)

        return shifted_eeg_data
