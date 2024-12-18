import torch
import torch.nn.functional as F

class AddVariousNoises(torch.nn.Module):
    """This class adds various types of noises to the input signal.

    Arguments
    ---------
    noise_types : list
        List of noise types to add, e.g., ['gaussian', 'poisson', 'salt_and_pepper'].
    noise_params : dict
        Dictionary of noise parameters for each type, e.g., {'gaussian': {'mean': 0, 'std': 0.1}, 'poisson': {}, 'salt_and_pepper': {'prob': 0.05}}.
    """

    def __init__(self, noise_types, noise_params):
        super().__init__()

        self.noise_types = noise_types
        self.noise_params = noise_params

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be [batch, time] or [batch, time, channels].

        Returns
        -------
        Tensor of shape [batch, time] or [batch, time, channels].
        """
        noisy_waveform = waveforms.clone()

        for noise_type in self.noise_types:
            if noise_type == 'gaussian':
                mean = self.noise_params.get(noise_type, {}).get('mean', 0)
                std = self.noise_params.get(noise_type, {}).get('std', 0.1)
                noise = torch.randn_like(waveforms) * std + mean

            elif noise_type == 'poisson':
                # Ensure that the waveforms are non-negative before applying Poisson noise
                non_negative_waveforms = torch.abs(waveforms)
                noise = torch.poisson(non_negative_waveforms)

            elif noise_type == 'salt_and_pepper':
                prob = self.noise_params.get(noise_type, {}).get('prob', 0.05)
                noise = F.dropout(waveforms, p=prob, training=self.training, inplace=False)

            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            noisy_waveform += noise

        return noisy_waveform
