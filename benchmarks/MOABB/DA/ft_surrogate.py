import numpy as np
import torch

def ft_surrogate(eeg_data, lambda_=1.0, probability=0.5):
    """
    Applies Fourier Transform Surrogate (FTS) augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - lambda_: Phase noise magnitude (between 0 and 1)
    - probability: Probability of applying the FTS to each window

    Returns:
    - surrogate_eeg_data: Surrogate EEG data as a PyTorch tensor
    """
    eeg_data_np = eeg_data.numpy()  # Convert to numpy for processing
    surrogate_eeg_data_np = np.zeros_like(eeg_data_np)

    for sample_idx, sample in enumerate(eeg_data_np):
        for channel_idx, channel in enumerate(sample.T):  # Transpose to iterate over channels
            if np.random.rand() < probability:
                # Perform Fourier transform
                fft_coeffs = np.fft.fft(channel)
                amplitudes = np.abs(fft_coeffs)
                phases = np.angle(fft_coeffs)
                
                # Generate new random phases
                new_phases = np.random.uniform(0, lambda_ * 2 * np.pi, size=phases.shape)
                
                # Construct new Fourier coefficients with original amplitudes and new phases
                new_fft_coeffs = amplitudes * np.exp(1j * new_phases)
                
                # Perform inverse Fourier transform to get the surrogate signal
                surrogate_channel = np.fft.ifft(new_fft_coeffs).real
                
                surrogate_eeg_data_np[sample_idx, :, channel_idx] = surrogate_channel
            else:
                surrogate_eeg_data_np[sample_idx, :, channel_idx] = channel

    surrogate_eeg_data = torch.from_numpy(surrogate_eeg_data_np).float()  # Convert back to PyTorch tensor

    return surrogate_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
surrogate_train_data = ft_surrogate(train_data, lambda_=1.0, probability=0.5)

# Print original and surrogate data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Surrogate Data (Sample 0, Channel 0):", surrogate_train_data[0, :, 0])
