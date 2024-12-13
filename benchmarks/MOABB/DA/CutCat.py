import random

import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    convolve1d,
    dB_to_amplitude,
    notch_filter,
    reverberate,
)

class CutCat(torch.nn.Module):
    """This function combines segments (with equal length in time) of the time series contained in the batch.
    Proposed for EEG signals in https://doi.org/10.1016/j.neunet.2021.05.032.

    Arguments
    ---------
    min_num_segments : int
        The number of segments to combine.
    max_num_segments : int
        The maximum number of segments to combine. Default is 10.

    Example
    -------
    >>> signal = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1,))
    >>> cutcat =  CutCat()
    >>> output_signal = cutcat(signal)
    """

    def __init__(self, min_num_segments=2, max_num_segments=10):
        super().__init__()
        self.min_num_segments = min_num_segments
        self.max_num_segments = max_num_segments
        # Check arguments
        if self.max_num_segments < self.min_num_segments:
            raise ValueError("max_num_segments must be  >= min_num_segments")

    def forward(self, waveforms):

        print("SHAPE")
        print(waveforms.shape)
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        if (
            waveforms.shape[0] > 1
        ):  # only if there are at least 2 examples in batch
            # rolling waveforms to point to segments of other examples in batch
            waveforms_rolled = torch.roll(waveforms, shifts=1, dims=0)
            # picking number of segments to use
            num_segments = torch.randint(
                low=self.min_num_segments,
                high=self.max_num_segments + 1,
                size=(1,),
            )
            # index of cuts (both starts and stops)
            idx_cut = torch.linspace(
                0, waveforms.shape[1], num_segments.item() + 1, dtype=torch.int
            )
            for i in range(idx_cut.shape[0] - 1):
                # half of segments from other examples in batch
                if i % 2 == 1:
                    start = idx_cut[i]
                    stop = idx_cut[i + 1]
                    waveforms[:, start:stop, ...] = waveforms_rolled[
                        :, start:stop, ...  # noqa: W504
                    ]

        return waveforms