# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:26:24 2023

@author: flori
"""

import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
import itertools


class ISDataset(data.Dataset):
    def __init__(self, if_x: np.array, if_y: np.array, smoothing: bool, normalize: bool, coarse_dropout: bool, permute: str):
        self.if_x = if_x
        self.if_y = if_y
        self.smoothing = smoothing
        self.normalize = normalize
        if permute == "full":
            # OLD METHOD => permuting everything
            n_channels = self.if_x.shape[-1]  # how many channels total (note: the first channel will never be permuted, since it's ELP)
            channels_to_permute = np.arange(1, n_channels)  # which channels will be permuted? (not the first)
            # precompute permutations
            self.permutations = np.array(list(itertools.permutations(channels_to_permute)))
        elif permute == "gam_kl":
            # NEW METHOD => permute GAM and KL separately
            gam_to_permute = [1, 2, 3]
            kl_to_permute = [4, 5]
            gam_permutations = np.array(list(itertools.permutations(gam_to_permute)))
            kl_permutations = np.array(list(itertools.permutations(kl_to_permute)))
            self.permutations = np.array([np.concatenate([gam, kl]) for gam in gam_permutations for kl in kl_permutations])
        elif permute is not None:
            assert False, f"Unhandled non-None {permute=}"
        else:
            self.permutations = None
        self.coarse_dropout = coarse_dropout

    def __len__(self):
        if self.permutations is not None:
            return len(self.permutations) * len(self.if_x)
        return len(self.if_x)

    def __getitem__(self, idx):
        if self.permutations is not None:
            item_idx = idx // len(self.permutations)  # which sample to select
            perm_idx = idx % len(self.permutations)  # which permutation to perform
            x = self.if_x[item_idx]
            y = self.if_y[item_idx]
            # permute channels
            permutation = self.permutations[perm_idx]
            x = x[:, [0, *permutation]]  # x will be 0 + ... (always ELP then permuted channels)
            y = y[:, permutation - 1]  # y will be ... (no ELP channel); note: channels have to be subtracted by 1 before since no "first channel" (ELP)
        else:
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
