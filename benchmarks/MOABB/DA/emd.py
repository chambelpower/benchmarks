import numpy as np
import torch
import scipy.interpolate as si
import matplotlib.pyplot as plt

def find_extrema(signal):
    """
    Find local maxima and minima of a signal.
    """
    d = np.diff(np.sign(np.diff(signal)))
    maxima = np.where(d < 0)[0] + 1
    minima = np.where(d > 0)[0] + 1
    return maxima, minima

def compute_envelopes(signal, maxima, minima):
    """
    Compute the upper and lower envelopes of a signal.
    """
    t = np.arange(len(signal))
    upper_env = si.interp1d(maxima, signal[maxima], kind='cubic', fill_value='extrapolate')(t)
    lower_env = si.interp1d(minima, signal[minima], kind='cubic', fill_value='extrapolate')(t)
    return upper_env, lower_env

def is_imf(signal, upper_env, lower_env, threshold=0.05):
    """
    Check if the signal is an IMF.
    """
    mean_env = (upper_env + lower_env) / 2
    std_dev = np.std(signal - mean_env)
    return std_dev < threshold

def emd(signal, max_imfs=15, max_sifts=10):
    """
    Perform Empirical Mode Decomposition on a signal.
    """
    imfs = []
    residual = signal
    
    for i in range(max_imfs):
        h = residual
        for j in range(max_sifts):
            maxima, minima = find_extrema(h)
            
            if len(maxima) < 2 or len(minima) < 2:
                break
            
            upper_env, lower_env = compute_envelopes(h, maxima, minima)
            mean_env = (upper_env + lower_env) / 2
            
            h = h - mean_env
            
            if is_imf(h, upper_env, lower_env):
                break
        
        imfs.append(h)
        residual = residual - h
        
        if np.all(np.abs(residual) < 1e-10) or len(find_extrema(residual)[0]) < 2:
            break
    
    # Pad IMFs with zeros if less than max_imfs
    while len(imfs) < max_imfs:
        imfs.append(np.zeros_like(signal))
    
    return np.array(imfs)

def decompose_eeg_frames(eeg_frames):
    """
    Decompose each EEG frame into its IMFs.
    """
    decomposed_frames = []
    for frame in eeg_frames:
        decomposed_channels = []
        for channel in frame:
            imfs = emd(channel.numpy())
            decomposed_channels.append(imfs)
        decomposed_frames.append(np.array(decomposed_channels))
    return np.array(decomposed_frames)

def generate_artificial_eeg_frames(eeg_frames, num_artificial_frames=15):
    """
    Generate artificial EEG frames using IMFs from the original frames.
    """
    decomposed_frames = decompose_eeg_frames(eeg_frames)
    
    num_channels = decomposed_frames.shape[1]
    num_samples = decomposed_frames.shape[3]
    num_imfs = decomposed_frames.shape[2]
    
    artificial_frames = []
    for _ in range(num_artificial_frames):
        artificial_frame = np.zeros((num_samples, num_channels))
        for imf_idx in range(num_imfs):
            frame_idx = np.random.randint(0, decomposed_frames.shape[0])
            artificial_frame += decomposed_frames[frame_idx, :, imf_idx, :].T
        artificial_frames.append(artificial_frame)
    
    return np.array(artificial_frames)

# Example usage with dummy EEG data
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))

# Generate artificial frames
artificial_frames = generate_artificial_eeg_frames(train_data, num_artificial_frames=15)

# Print or plot the artificial frames
print(artificial_frames.shape)
plt.plot(artificial_frames[0, :, 0])
plt.show()
