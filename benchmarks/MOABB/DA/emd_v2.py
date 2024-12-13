import torch
import torch.nn as nn
import scipy.interpolate as si
import numpy as np

class EMDGenerator(nn.Module):
    def __init__(self, max_imfs=15, max_sifts=10, num_artificial_frames=15):
        super(EMDGenerator, self).__init__()
        self.max_imfs = max_imfs
        self.max_sifts = max_sifts
        self.num_artificial_frames = num_artificial_frames
    
    def find_extrema(self, signal):
        d = np.diff(np.sign(np.diff(signal)))
        maxima = np.where(d < 0)[0] + 1
        minima = np.where(d > 0)[0] + 1
        return maxima, minima

    def compute_envelopes(self, signal, maxima, minima):
        t = np.arange(len(signal))
        
        if len(maxima) < 4 or len(minima) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'
        
        if len(maxima) > 1:
            upper_env = si.interp1d(maxima, signal[maxima], kind=kind, fill_value='extrapolate')(t)
        else:
            upper_env = np.zeros_like(signal)
        
        if len(minima) > 1:
            lower_env = si.interp1d(minima, signal[minima], kind=kind, fill_value='extrapolate')(t)
        else:
            lower_env = np.zeros_like(signal)
        
        return upper_env, lower_env

    def is_imf(self, signal, upper_env, lower_env, threshold=0.05):
        mean_env = (upper_env + lower_env) / 2
        std_dev = np.std(signal - mean_env)
        return std_dev < threshold

    def emd(self, signal):
        imfs = []
        residual = signal

        for i in range(self.max_imfs):
            h = residual
            for j in range(self.max_sifts):
                maxima, minima = self.find_extrema(h)
                
                if len(maxima) < 2 or len(minima) < 2:
                    break
                
                upper_env, lower_env = self.compute_envelopes(h, maxima, minima)
                mean_env = (upper_env + lower_env) / 2
                
                h = h - mean_env
                
                if self.is_imf(h, upper_env, lower_env):
                    break
            
            imfs.append(h)
            residual = residual - h
            
            if np.all(np.abs(residual) < 1e-10) or len(self.find_extrema(residual)[0]) < 2:
                break
        
        while len(imfs) < self.max_imfs:
            imfs.append(np.zeros_like(signal))
        
        return np.array(imfs)

    def decompose_eeg_frames(self, eeg_frames):
        decomposed_frames = []
        for frame in eeg_frames:
            decomposed_channels = []
            for channel in frame:
                imfs = self.emd(channel.cpu().numpy())
                decomposed_channels.append(imfs)
            decomposed_frames.append(np.array(decomposed_channels))
        
        # Convert the list of NumPy arrays to a single NumPy array before converting to a tensor
        decomposed_frames_np = np.array(decomposed_frames)
        return torch.tensor(decomposed_frames_np, dtype=eeg_frames.dtype)

    def forward(self, waveforms):
        # Get the batch size of the input
        original_batch_size = waveforms.shape[0]
        self.num_artificial_frames = original_batch_size
        
        # Decompose the waveforms into frames
        decomposed_frames = self.decompose_eeg_frames(waveforms)
        
        num_channels = decomposed_frames.shape[1]
        num_samples = decomposed_frames.shape[3]
        num_imfs = decomposed_frames.shape[2]
        
        artificial_frames = []
        for _ in range(self.num_artificial_frames):
            artificial_frame = torch.zeros((num_channels, num_samples), dtype=waveforms.dtype)
            for imf_idx in range(num_imfs):
                frame_idx = np.random.randint(0, decomposed_frames.shape[0])
                artificial_frame += decomposed_frames[frame_idx, :, imf_idx, :]
            artificial_frames.append(artificial_frame)
        
        # Ensuring that the number of artificial frames matches the original batch size
        artificial_frames = torch.stack(artificial_frames)[:original_batch_size]
        
        # Return a tensor with the same batch size as the original input
        return artificial_frames
