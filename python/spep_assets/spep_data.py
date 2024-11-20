# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:26:24 2023

@author: flori
"""

import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import gaussian_filter


class ISDataset(data.Dataset):
    def __init__(self, if_x: np.array, if_y: np.array, smoothing: bool, normalize: bool, coarse_dropout: bool):
        self.if_x = if_x
        self.if_y = if_y
        self.smoothing = smoothing
        self.normalize = normalize
        self.coarse_dropout = coarse_dropout

    def __len__(self):
        return len(self.if_x)

    def __getitem__(self, idx):
        x = self.if_x[idx]
        y = self.if_y[idx]
        # smooth
        if self.smoothing:
            x = gaussian_filter(x, sigma=3, axes=-1)  # apply gaussian filter with sigma=3 for smoothing
        # normalize
        if self.normalize:
            xmin = x.min(axis=1, keepdims=True)
            xmax = x.max(axis=1, keepdims=True)
            x = (x - xmin) / (xmax - xmin)
        # random coarse dropouts => we randomly put some wide areas to zero to simulate what would happen if we removed (dropped out) peak(s)
        if self.coarse_dropout:
            cd_params = dict(n_binom_n=10, n_binom_p=.5, width_binom_n=200, width_binom_p=.5)
            n_coarse_dropouts = np.random.binomial(cd_params["n_binom_n"], cd_params["n_binom_p"])
            if n_coarse_dropouts > 0:
                # randomly determine dropouts location, size and track
                coarse_dropouts_width = np.random.binomial(cd_params["width_binom_n"], cd_params["width_binom_p"], n_coarse_dropouts)
                coarse_dropouts_track = np.random.uniform(0, 5, n_coarse_dropouts).astype(int)
                coarse_dropouts_center = [np.random.choice(x.shape[-1], p=x[track] / x[track].sum()) for track in coarse_dropouts_track]
                # apply
                for coarse_loc, coarse_width, coarse_track in zip(coarse_dropouts_center, coarse_dropouts_width, coarse_dropouts_track):
                    x[coarse_track, (coarse_loc - coarse_width):(coarse_loc + coarse_width)] = 0
        # reshape according to pytorch standards => channels, height, width (should be 6, 1, 304)
        x = x.reshape((x.shape[-1], 1, x.shape[0], ))
        # turn to tensor
        x = torch.as_tensor(x, dtype=torch.float32)
        return x, y
