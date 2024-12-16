# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:26:24 2023

@author: flori
"""

# import importlib
# first_script=importlib.import_module("0_first_load_ms_data")

# from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed

# according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10353832/ et al., batch size should be kept low? but changing batch size during training is messy

# VAE from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py, modified


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as pl
from matplotlib import pyplot as plt
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def _get_homemade_encoder_v1_base_modules(input_dim, input_channels, encoder_blocks_layout, maxpool, kernel_size, stride, dropout,
                                          first_layer_groups, groups, batchnorm, hidden_activation, flatten):

    assert (maxpool == 1 and stride > 1) or (maxpool > 1 and stride == 1), f"Exactly one of {maxpool=} and {stride=} must be > 1"
    n_pools = len(encoder_blocks_layout) - 1 if maxpool > 1 else len(encoder_blocks_layout)

    _encoder_modules = []

    previous_channels = input_channels

    for block_position, block_layers in enumerate(encoder_blocks_layout):
        if block_position > 0:
            if maxpool > 1:
                _encoder_modules.append(nn.MaxPool2d(kernel_size=(1, maxpool), stride=(1, maxpool)))
        for layer_position, layer_channels in enumerate(block_layers):
            current_stride = stride if layer_position == 0 else 1
            padding = 0
            if current_stride > 1:
                hin = input_dim // (max(maxpool, stride) ** (layer_position))
                padding = (kernel_size - hin + stride * (hin / stride - 1)) / 2
                assert (padding % 1) == 0, f"Current {stride}, {hin=} and {kernel_size=} end with decimal {padding=}"
                padding = int(padding)

            _encoder_modules.append(nn.Conv2d(in_channels=previous_channels, out_channels=layer_channels, kernel_size=(1, kernel_size),
                                              stride=(1, current_stride),
                                              groups=first_layer_groups if ((layer_position == 0) and (block_position == 0)) else groups,
                                              padding=(0, padding) if current_stride > 1 else "same"))
            if (dropout is not None) and (dropout > 0):
                _encoder_modules.append(nn.Dropout(p=dropout))
            previous_channels = layer_channels
            if batchnorm:
                _encoder_modules.append(nn.BatchNorm2d(layer_channels))
            if hidden_activation == "relu":
                _encoder_modules.append(nn.ReLU())
            elif hidden_activation == "leaky_relu":
                _encoder_modules.append(nn.LeakyReLU())
            elif hidden_activation == "elu":
                _encoder_modules.append(nn.ELU())
            elif hidden_activation == "selu":
                _encoder_modules.append(nn.SELU())
            else:
                assert False, f"Unknown {hidden_activation=}"

    encoder_lastconv_dim = input_dim // (max(stride, maxpool) ** n_pools)
    encoder_lastconv_channels = encoder_blocks_layout[-1][-1]

    if flatten == "avg":
        global_pooling_kernel = encoder_lastconv_dim
        encoder_output_dim = encoder_lastconv_channels
        _encoder_modules.append(nn.AvgPool2d(kernel_size=(1, global_pooling_kernel)))
        _encoder_modules.append(nn.Flatten())
    elif flatten == "max":
        global_pooling_kernel = encoder_lastconv_dim
        encoder_output_dim = encoder_lastconv_channels
        _encoder_modules.append(nn.MaxPool2d(kernel_size=(1, global_pooling_kernel)))
        _encoder_modules.append(nn.Flatten())
    elif flatten == "flatten":
        encoder_output_dim = encoder_lastconv_channels * encoder_lastconv_dim
        _encoder_modules.append(nn.Flatten())
    else:
        assert False, f"Unknown {flatten=}"

    return _encoder_modules, encoder_output_dim  # , encoder_lastconv_channels, encoder_lastconv_dim


def _get_homemade_decoder_v1_base_modules(input_dim, latent_dim, include_initial_dense, input_channels, encoder_blocks_layout, maxpool,
                                          kernel_size, stride, first_layer_groups, groups, batchnorm, output_activation, hidden_activation):

    assert (maxpool == 1 and stride > 1) or (maxpool > 1 and stride == 1), f"Exactly one of {maxpool=} and {stride=} must be > 1"
    stride_pool = max(maxpool, stride)
    n_pools = len(encoder_blocks_layout) - 1 if maxpool > 1 else len(encoder_blocks_layout)

    _decoder_modules = []

    decoder_firstconv_dim = input_dim // (stride_pool ** n_pools)
    decoder_firstconv_channels = encoder_blocks_layout[-1][-1]
    decoder_input_dim = decoder_firstconv_dim * decoder_firstconv_channels

    if include_initial_dense and (latent_dim > 0):
        _decoder_modules.append(nn.Linear(latent_dim, decoder_input_dim))
    _decoder_modules.append(nn.Unflatten(1, (decoder_firstconv_channels, 1, decoder_firstconv_dim)))

    previous_channels = decoder_firstconv_channels

    start_blocks_at = 1 if maxpool > 1 else 0

    for block_position, block_layers in enumerate(encoder_blocks_layout[::-1][start_blocks_at:]):
        for layer_position, layer_channels in enumerate(block_layers[::-1]):
            if layer_position == 0:
                _decoder_modules.append(nn.ConvTranspose2d(in_channels=previous_channels, out_channels=layer_channels, kernel_size=(1, stride_pool),
                                                           stride=(1, stride_pool), groups=first_layer_groups if block_position == 0 else groups))
            else:
                _decoder_modules.append(nn.Conv2d(in_channels=previous_channels, out_channels=layer_channels, kernel_size=(1, kernel_size), stride=(1, 1),
                                                  groups=groups, padding="same"))
            previous_channels = layer_channels
            if batchnorm:
                _decoder_modules.append(nn.BatchNorm2d(layer_channels))
            if hidden_activation == "relu":
                _decoder_modules.append(nn.ReLU())
            elif hidden_activation == "leaky_relu":
                _decoder_modules.append(nn.LeakyReLU())
            elif hidden_activation == "elu":
                _decoder_modules.append(nn.ELU())
            elif hidden_activation == "selu":
                _decoder_modules.append(nn.SELU())
            else:
                assert False, f"Unknown {hidden_activation=}"

    _decoder_modules.append(nn.Conv2d(in_channels=previous_channels, out_channels=input_channels, kernel_size=(1, kernel_size), stride=(1, 1),
                                      groups=groups, padding="same"))
    if output_activation == "sigmoid":
        _decoder_modules.append(nn.Sigmoid())
    elif output_activation != "none":
        assert False, f"Unknown {output_activation=}"

    return _decoder_modules


def _get_dense_encoder(input_dim, input_channels, encoder_blocks_layout, dropout, batchnorm, hidden_activation):

    _encoder_modules = []

    previous_size = input_dim * input_channels

    _encoder_modules.append(nn.Flatten())

    for layer_position, layer_size in enumerate(encoder_blocks_layout):

        _encoder_modules.append(nn.Linear(previous_size, layer_size))

        if (dropout is not None) and (dropout > 0):
            _encoder_modules.append(nn.Dropout(p=dropout))
        previous_size = layer_size
        if batchnorm:
            _encoder_modules.append(nn.BatchNorm1d(layer_size))
        if hidden_activation == "relu":
            _encoder_modules.append(nn.ReLU())
        elif hidden_activation == "leaky_relu":
            _encoder_modules.append(nn.LeakyReLU())
        elif hidden_activation == "elu":
            _encoder_modules.append(nn.ELU())
        elif hidden_activation == "selu":
            _encoder_modules.append(nn.SELU())
        else:
            assert False, f"Unknown {hidden_activation=}"

    return _encoder_modules, previous_size


def _plot_autoencoded_samples(x_input_np, x_output_np, y_np, title_left=None, title_right=None):
    from matplotlib import gridspec
    possible_y_classes = np.unique(y_np).tolist()
    n_samples = len(possible_y_classes)
    plt.figure(figsize=(n_samples * 5, 5 * 3))
    gs = gridspec.GridSpec(5, n_samples)
    for y_i, current_y in enumerate(possible_y_classes):
        idx = np.where(y_np == current_y)[0][0]
        for track in range(5):
            sample_input = x_input_np[idx, track, 0, :]
            xmin = np.min(sample_input)
            xmax = np.max(sample_input)
            xrange = xmax - xmin
            xmin = xmin - .05 * xrange
            xmax = xmax + .05 * xrange
            sample_output = x_output_np[idx, track, 0, :]
            # subplot_i = y_i + n_samples * track + 1
            # plt.subplot(5, n_samples, subplot_i)
            residuals = ((sample_output - sample_input) + 1) / 2
            plt.subplot(gs[track, y_i])
            plt.plot(sample_input, color="#aaaaaa")
            plt.plot(sample_output, color="black")
            plt.plot(residuals, color="red")
            plt.text(0, sample_input[0] + .02, "input", color="#aaaaaa", horizontalalignment="left", verticalalignment="bottom", size=8)
            plt.text(len(sample_output), sample_output[0] + .02, "output", color="black", horizontalalignment="right", verticalalignment="bottom", size=8)
            plt.text(0, residuals[0] + .02, "residuals", color="red", horizontalalignment="left", verticalalignment="bottom", size=8)
            plt.ylim((xmin, xmax))
            plt.xlim((0, sample_input.shape[-1]))
            if track == 4:
                if (y_i == 0) and (title_left is not None):
                    plt.title(title_left)
                elif (y_i == len(possible_y_classes) - 1) and (title_right is not None):
                    plt.title(title_right)
    plt.tight_layout()


class AE(pl.LightningModule):
    def __init__(self, backbone, input_dim, input_channels, latent_dim, kernel_size, encoder_blocks_layout, flatten,
                 stride, dropout, groups, first_layer_groups, maxpool, batchnorm, hidden_activation, output_activation,
                 lr, lr_scheduler, lr_multistep_milestones, lr_multistep_gamma,
                 probe_model, probe_cv, output_images_path):
        super().__init__()
        self.output_images_path = output_images_path
        os.makedirs(self.output_images_path, exist_ok=True)
        self.validation_x_input = []
        self.validation_encodings = []
        self.validation_x_output = []
        self.validation_y = []

        assert backbone == "homemade", f"Autoencoder does not support {backbone=} yet"
        assert flatten == "flatten", f"Autoencoder does not support {flatten=} yet"

        if probe_model == "svm-rbf":
            self.probe_model = svm.SVC(kernel="rbf")
        elif probe_model == "svm-linear":
            self.probe_model = svm.SVC(kernel="linear")
        elif probe_model == "ridge":
            self.probe_model = LogisticRegression(penalty="l2")
        elif probe_model == "lasso":
            self.probe_model = LogisticRegression(penalty="l1")
        else:
            assert False, f"Unknown {probe_model=}"
        self.probe_cv = probe_cv

        # prev_channels = input_channels
        # self.enc_out_dim = input_dim * channels // 2

        if backbone == "homemade":
            encoder_modules, enc_out_dim = _get_homemade_encoder_v1_base_modules(input_dim=input_dim, input_channels=input_channels, maxpool=maxpool,
                                                                                 encoder_blocks_layout=encoder_blocks_layout,
                                                                                 kernel_size=kernel_size, stride=stride, dropout=dropout,
                                                                                 first_layer_groups=first_layer_groups, groups=groups, flatten=flatten,
                                                                                 batchnorm=batchnorm, hidden_activation=hidden_activation)

            decoder_modules = _get_homemade_decoder_v1_base_modules(input_dim=input_dim, latent_dim=latent_dim, include_initial_dense=latent_dim>0,
                                                                    input_channels=input_channels, encoder_blocks_layout=encoder_blocks_layout,
                                                                    maxpool=maxpool, kernel_size=kernel_size,
                                                                    stride=stride, first_layer_groups=first_layer_groups, groups=groups,
                                                                    batchnorm=batchnorm, hidden_activation=hidden_activation,
                                                                    output_activation=output_activation)

            self.encoder = nn.Sequential(*encoder_modules)
            self.decoder = nn.Sequential(*decoder_modules)
            self.enc_out_dim = enc_out_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.lr_scheduler == "multistep":
            print(f"Setting optimizer to multistep with params {self.hparams.lr_multistep_milestones=}, {self.hparams.lr_multistep_gamma=}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_multistep_milestones, gamma=self.hparams.lr_multistep_gamma,
                                                             verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.hparams.lr_scheduler != "none":
            assert False, f"Unknown/incompatible {self.hparams.lr_scheduler=}"
        return optimizer

    def on_validation_epoch_end(self):
        if len(self.validation_x_input) > 0:

            x_input = torch.cat(self.validation_x_input, dim=0).detach().cpu().numpy()
            encodings = torch.cat(self.validation_encodings, dim=0).detach().cpu().numpy()
            x_output = torch.cat(self.validation_x_output, dim=0).detach().cpu().numpy()
            y = torch.cat(self.validation_y, dim=0).detach().cpu().numpy()

            # validation step
            possible_y_classes = np.unique(y).tolist()

            mean_score = np.nan
            if len(possible_y_classes) > 1:  # at least 2 classes
                mean_scores = []
                for left_i, left_y in enumerate(possible_y_classes):
                    for right_y in possible_y_classes[left_i + 1:]:
                        probe_cv = self.probe_cv
                        least_class_n = min(np.sum(y == left_y), np.sum(y == right_y))
                        if least_class_n < probe_cv:  # at least N elements per class, with N >= cv
                            probe_cv = least_class_n
                        filt = (y == left_y) | (y == right_y)
                        cv_outputs = cross_validate(estimator=self.probe_model,
                                                    X=encodings[filt],
                                                    y=(y[filt] == left_y) * 1,
                                                    scoring='accuracy',
                                                    cv=probe_cv)
                        mean_score = cv_outputs['test_score'].mean()
                        mean_scores.append(mean_score)
                mean_score = np.mean(mean_scores)

            # reconstruction metric
            xmin = x_input.min(axis=(2, 3), keepdims=True)
            xmax = x_input.max(axis=(2, 3), keepdims=True)
            x_input_norm = (x_input - xmin) / (xmax - xmin)
            x_output_norm = (x_output - xmin) / (xmax - xmin)
            rec_mae = np.abs(x_input_norm - x_output_norm).mean()  # L1 reconstruction error

            # n_samples = len(possible_y_classes)
            epoch = self.current_epoch
            step = self.global_step
            title_left = f"{epoch=: 4d} {step=: 8d}"
            title_right = f"{mean_score=:.4f} {rec_mae=:.4f}"
            _plot_autoencoded_samples(x_input_np=x_input, x_output_np=x_output, y_np=y, title_left=title_left, title_right=title_right)
            plt.savefig(os.path.join(self.output_images_path, f"epoch={self.current_epoch}-step={self.global_step}.png"))
            plt.close()

            self.validation_x_input.clear()
            self.validation_encodings.clear()
            self.validation_x_output.clear()
            self.validation_y.clear()


def gaussian_likelihood(x_hat, x, logscale):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


def monte_carlo_kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl


def kl_divergence(mu, logvar):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss


class VAE(AE):
    def __init__(self, backbone,
                 input_dim, input_channels, latent_dim, kernel_size, encoder_blocks_layout, flatten,
                 stride, dropout, groups, first_layer_groups, maxpool, batchnorm, hidden_activation, output_activation,
                 lr, lr_scheduler, lr_multistep_milestones, lr_multistep_gamma,
                 reconstruction_loss, regularization_loss_lambda,
                 probe_model, probe_cv, output_images_path):
        super().__init__(backbone=backbone, input_dim=input_dim, input_channels=input_channels, latent_dim=latent_dim, kernel_size=kernel_size,
                         encoder_blocks_layout=encoder_blocks_layout, flatten=flatten, stride=stride, dropout=dropout,
                         groups=groups, first_layer_groups=first_layer_groups, maxpool=maxpool, batchnorm=batchnorm,
                         hidden_activation=hidden_activation, output_activation=output_activation,
                         lr=lr, lr_scheduler=lr_scheduler, lr_multistep_milestones=lr_multistep_milestones, lr_multistep_gamma=lr_multistep_gamma,
                         probe_model=probe_model, probe_cv=probe_cv,
                         output_images_path=output_images_path[0] if type(output_images_path) is list else output_images_path)
        self.save_hyperparameters()

        if type(output_images_path) is list:
            self.log_val_at_epoch = output_images_path[1]
        else:
            self.log_val_at_epoch = None

        assert latent_dim > 0, f"Unsupported {latent_dim=}"

        # distribution parameters
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # now compatible with loss weights
        if reconstruction_loss == "gaussian_likelihood":
            self.logscale = nn.Parameter(torch.Tensor([0.0]))
            self.reconstruction_loss = lambda x_hat, x: gaussian_likelihood(x_hat, x, self.logscale)
        elif reconstruction_loss == "mse":
            # self.reconstruction_loss = F.mse_loss
            self.reconstruction_loss = lambda *kwargs: F.mse_loss(*kwargs, reduction='none')
        elif reconstruction_loss == "mae":
            # self.reconstruction_loss = F.l1_loss
            self.reconstruction_loss = lambda *kwargs: F.l1_loss(*kwargs, reduction='none')
        elif reconstruction_loss == "bce":
            # self.reconstruction_loss = torch.nn.BCELoss()
            self.reconstruction_loss = torch.nn.BCELoss(reduction='none')
        else:
            assert False, f"Unknown/unsupported {reconstruction_loss=}"

        self.regularization_loss_lambda = regularization_loss_lambda

    def forward(self, batch: list) -> tuple:
        x, *_ = batch
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, z, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def training_step(self, batch, batch_idx):
        # x, *_ = batch
        x, weights = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, logvar = self.fc_mu(x_encoded), self.fc_logvar(x_encoded)

        # sample z from q
        z = self.reparameterize(mu, logvar)

        # decoded
        x_hat = self.decoder(z)

        # old recon loss
        # recon_loss = self.reconstruction_loss(x_hat, x)

        # compatible with loss weights
        recon_loss = self.reconstruction_loss(x_hat, x)
        if len(recon_loss.shape) > 1:
            recon_loss = recon_loss.mean(axis=(1, 2, 3))
        recon_loss = recon_loss * weights
        recon_loss = recon_loss.mean()

        kld_loss = self.regularization_loss_lambda * kl_divergence(mu, logvar)

        total_loss = recon_loss + kld_loss

        # if False:
        #     idx = 1
        #     plt.figure(figsize=(16,16))
        #     for i in range(5):
        #         x_real = x[idx, i, 0, :].detach().cpu().numpy()
        #         x_recon = x_hat[idx, i, 0, :].detach().cpu().numpy()
        #         plt.subplot(5, 1, i+1)
        #         plt.plot(x_real, color="#aaaaaa")
        #         plt.plot(x_recon, color="black")
        #         plt.plot(((x_recon - x_real) + 1) / 2, color="red")
        #     plt.tight_layout()
        #     plt.show()

        self.log_dict({
            'total': total_loss,
            'kl': kld_loss,
            'recon': recon_loss,
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        if (self.log_val_at_epoch is not None) and (self.current_epoch not in self.log_val_at_epoch):
            return

        x, y = batch
        xhat = self.encoder(x)
        encodings = self.fc_mu(xhat)
        xhat = self.decoder(encodings)

        xhat, z, mu, logvar = self.forward(batch)

        self.validation_x_input.append(x)
        self.validation_encodings.append(encodings)
        self.validation_x_output.append(xhat)
        self.validation_y.append(y)


class SAE(AE):
    def __init__(self, backbone,
                 input_dim, input_channels, latent_dim, kernel_size, encoder_blocks_layout, flatten,
                 stride, dropout, groups, first_layer_groups, maxpool, batchnorm, hidden_activation, output_activation,
                 lr, lr_scheduler, lr_multistep_milestones, lr_multistep_gamma,
                 reconstruction_loss, regularization_loss, regularization_loss_lambda, regularization_loss_bias,
                 probe_model, probe_cv, output_images_path):
        output_images_path[0] if type(output_images_path) is list else output_images_path

        super().__init__(backbone=backbone, input_dim=input_dim, input_channels=input_channels, latent_dim=latent_dim, kernel_size=kernel_size,
                         encoder_blocks_layout=encoder_blocks_layout, flatten=flatten, stride=stride, dropout=dropout,
                         groups=groups, first_layer_groups=first_layer_groups, maxpool=maxpool, batchnorm=batchnorm,
                         hidden_activation=hidden_activation, output_activation=output_activation,
                         lr=lr, lr_scheduler=lr_scheduler, lr_multistep_milestones=lr_multistep_milestones, lr_multistep_gamma=lr_multistep_gamma,
                         probe_model=probe_model, probe_cv=probe_cv,
                         output_images_path=output_images_path[0] if type(output_images_path) is list else output_images_path)
        self.save_hyperparameters()

        if type(output_images_path) is list:
            self.log_val_at_epoch = output_images_path[1]
        else:
            self.log_val_at_epoch = None

        # fc
        if latent_dim > 0:
            self.fc = nn.Linear(self.enc_out_dim, latent_dim)
        else:
            self.fc = None

        # now compatible with loss weights
        if reconstruction_loss == "gaussian_likelihood":
            self.logscale = nn.Parameter(torch.Tensor([0.0]))
            self.reconstruction_loss = lambda x_hat, x: gaussian_likelihood(x_hat, x, self.logscale)
        elif reconstruction_loss == "mse":
            # self.reconstruction_loss = F.mse_loss
            self.reconstruction_loss = lambda *kwargs: F.mse_loss(*kwargs, reduction='none')
        elif reconstruction_loss == "mae":
            # self.reconstruction_loss = F.l1_loss
            self.reconstruction_loss = lambda *kwargs: F.l1_loss(*kwargs, reduction='none')
        elif reconstruction_loss == "bce":
            # self.reconstruction_loss = torch.nn.BCELoss()
            self.reconstruction_loss = torch.nn.BCELoss(reduction='none')
        else:
            assert False, f"Unknown/unsupported {reconstruction_loss=}"

        if regularization_loss == "l1":
            self.regularization_loss = lambda *kwargs: F.l1_loss(*kwargs, reduction='sum')  # compute sum instead of mean
            # self.regularization_loss = nn.L1Loss()
        elif regularization_loss == "l2":
            self.regularization_loss = lambda *kwargs: F.mse_loss(*kwargs, reduction='sum')  # compute sum instead of mean
            # self.regularization_loss = RMSELoss()
        else:
            assert False, f"Unknown {regularization_loss=}"

        self.regularization_loss_lambda = regularization_loss_lambda
        self.regularization_loss_bias = regularization_loss_bias

        # distribution parameters

    def decode(self, batch):
        conv_encodings = batch
        if self.fc is not None:
            fcn_encodings = self.fc(conv_encodings)
            xhat = self.decoder(fcn_encodings)
            return xhat
        else:
            xhat = self.decoder(conv_encodings)
            return xhat

    def forward(self, batch):
        x, *_ = batch
        conv_encodings = self.encoder(x)
        if self.fc is not None:
            fcn_encodings = self.fc(conv_encodings)
            xhat = self.decoder(fcn_encodings)
            return xhat, fcn_encodings, conv_encodings
        else:
            xhat = self.decoder(conv_encodings)
            return xhat, None, conv_encodings

    def training_step(self, batch, batch_idx):
        x, weights = batch

        # encode x to get the mu and variance parameters
        if self.fc is not None:
            encodings = self.fc(self.encoder(x))
        else:
            encodings = self.encoder(x)
        x_hat = self.decoder(encodings)

        # compatible with loss weights
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        if len(reconstruction_loss.shape) > 1:
            reconstruction_loss = reconstruction_loss.mean(axis=(1, 2, 3))
        reconstruction_loss = reconstruction_loss * weights
        reconstruction_loss = reconstruction_loss.mean()

        # NOTE: if we want to l1 regularize the weights instead of the activations!
        # all_linear1_params = torch.cat([x.view(-1) for x in model.fc.parameters()])
        # l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)

        regularization_loss = self.regularization_loss_lambda * self.regularization_loss(encodings, torch.zeros_like(encodings) + torch.ones_like(encodings) * self.regularization_loss_bias)
        total_loss = regularization_loss + reconstruction_loss

        self.log_dict({
            'total': total_loss,
            'regularization_loss': regularization_loss,
            'reconstruction_loss': reconstruction_loss,
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        if (self.log_val_at_epoch is not None) and (self.current_epoch not in self.log_val_at_epoch):
            return

        x, y = batch
        xhat, fcn_encodings, conv_encodings = self.forward(batch)

        self.validation_x_input.append(x)
        if fcn_encodings is not None:
            self.validation_encodings.append(fcn_encodings)
        else:
            self.validation_encodings.append(conv_encodings)
        self.validation_x_output.append(xhat)
        self.validation_y.append(y)


class SupervisedModule(pl.LightningModule):
    def __init__(self, n_classes, mode,
                 input_dim, input_channels, latent_dim, kernel_size, flatten, stride, dropout, groups, first_layer_groups, maxpool,
                 encoder_blocks_layout, backbone, batchnorm, hidden_activation,
                 optimizer,
                 lr, lr_scheduler, lr_reduceonplateau_factor, lr_reduceonplateau_patience, lr_reduceonplateau_threshold, lr_reduceonplateau_minlr,
                 lr_multistep_milestones,
                 lr_multistep_gamma,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.mode = mode
        self.backbone = backbone

        if self.backbone == "resnet1d":
            # resnet1d from https://github.com/hsd1503/resnet1d
            from resnet1d.resnet1d import ResNet1D
            self.encoder = ResNet1D(in_channels=input_channels,
                                    base_filters=encoder_blocks_layout[0][0],
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    groups=groups,
                                    n_block=len(encoder_blocks_layout),
                                    n_classes=n_classes)

        elif self.backbone == "net1d":
            from resnet1d.net1d import Net1D
            self.encoder = Net1D(
                in_channels=input_channels,
                base_filters=encoder_blocks_layout[0][0],
                ratio=1.0,
                filter_list=[f[0] for f in encoder_blocks_layout],
                m_blocks_list=[len(f) for f in encoder_blocks_layout],
                kernel_size=kernel_size,
                stride=stride,
                groups_width=groups,
                verbose=False,
                n_classes=n_classes)

        elif self.backbone == "homemade":
            encoder_modules, enc_out_dim = _get_homemade_encoder_v1_base_modules(input_dim=input_dim, input_channels=input_channels, maxpool=maxpool,
                                                                                 encoder_blocks_layout=encoder_blocks_layout, kernel_size=kernel_size,
                                                                                 stride=stride, dropout=dropout,
                                                                                 first_layer_groups=first_layer_groups, groups=groups,
                                                                                 batchnorm=batchnorm, hidden_activation=hidden_activation, flatten=flatten)
            self.encoder = nn.Sequential(*encoder_modules)
            self.fc = nn.Linear(enc_out_dim, latent_dim)
            self.pred = nn.Linear(latent_dim, self.n_classes)

        elif self.backbone == "dense":
            encoder_modules, enc_out_dim = _get_dense_encoder(input_dim=input_dim, input_channels=input_channels, encoder_blocks_layout=encoder_blocks_layout,
                                                              dropout=dropout, batchnorm=batchnorm, hidden_activation=hidden_activation)
            self.encoder = nn.Sequential(*encoder_modules)
            # self.fc = nn.Linear(enc_out_dim, latent_dim)
            # self.pred = nn.Linear(latent_dim, self.n_classes)
            self.pred = nn.Linear(enc_out_dim, self.n_classes)
        elif self.backbone == "vit_1d":
            # based on
            # transformer implementation: vit_1d or simple_vit_1d from:
            # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_1d.py

            # hyperparameters: derived from ecg classifier from:
            # https://yonigottesman.github.io/ecg/vit/deep-learning/2023/01/20/ecg-vit.html

            # training implementation from:
            # https://hackmd.io/@arkel23/ryjgQ7p8u#Example-Training-with-PT-Trainer-and-Logging-with-WampB

            from vit_pytorch import vit_1d

            # for vit we used config parameter "encoder_blocks_layout" in a weird way:
            patch_size = encoder_blocks_layout[0]  # the patch size // defaults 16 from ViT, 20 from ecg classifier
            heads = encoder_blocks_layout[1]  # the number of self attention heads // defaults 6
            depth = encoder_blocks_layout[2]  # the number of layers (depth) // defaults 6
            mlp_dim = encoder_blocks_layout[3]  # the mlp dim // defaults 256
            dim_head = encoder_blocks_layout[4]  # the head dim // defaults 64
            hidden_dim = latent_dim  # the hidden_dim // defaults 768
            # dropout  // defaults 0.1

            vit = vit_1d.ViT(
                seq_len=input_dim,
                patch_size=patch_size,
                num_classes=self.n_classes,
                channels=input_channels,
                dim=hidden_dim,
                dim_head=dim_head,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                emb_dropout=dropout
            )

            self.encoder = vit
        elif self.backbone == "simple_vit_1d":
            # based on
            # transformer implementation: vit_1d or simple_vit_1d from:
            # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_1d.py

            # hyperparameters: derived from ecg classifier from:
            # https://yonigottesman.github.io/ecg/vit/deep-learning/2023/01/20/ecg-vit.html

            # training implementation from:
            # https://hackmd.io/@arkel23/ryjgQ7p8u#Example-Training-with-PT-Trainer-and-Logging-with-WampB

            from vit_pytorch import simple_vit_1d

            # for vit we used config parameter "encoder_blocks_layout" in a weird way:
            patch_size = encoder_blocks_layout[0]  # the patch size // defaults 16 from ViT, 20 from ecg classifier
            heads = encoder_blocks_layout[1]  # the number of self attention heads // defaults 6
            depth = encoder_blocks_layout[2]  # the number of layers (depth) // defaults 6
            mlp_dim = encoder_blocks_layout[3]  # the mlp dim // defaults 256
            dim_head = encoder_blocks_layout[4]  # the head dim // defaults 64
            hidden_dim = latent_dim  # the hidden_dim // defaults 768
            # dropout  // NO DROPOUT IN SIMPLE VIT!

            vit = simple_vit_1d.SimpleViT(
                seq_len=input_dim,
                patch_size=patch_size,
                num_classes=self.n_classes,
                channels=input_channels,
                dim=hidden_dim,
                dim_head=dim_head,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
            )

            self.encoder = vit

        elif self.backbone == "smalldataset_vit_1d":
            from python.spep_assets.vit_1d_for_small_dataset import SDViT

            # for vit we used config parameter "encoder_blocks_layout" in a weird way:
            patch_size = encoder_blocks_layout[0]  # the patch size // defaults 16 from ViT, 20 from ecg classifier
            heads = encoder_blocks_layout[1]  # the number of self attention heads // defaults 6
            depth = encoder_blocks_layout[2]  # the number of layers (depth) // defaults 6
            mlp_dim = encoder_blocks_layout[3]  # the mlp dim // defaults 256
            dim_head = encoder_blocks_layout[4]  # the head dim // defaults 64
            hidden_dim = latent_dim  # the hidden_dim // defaults 768
            # dropout  // NO DROPOUT IN SIMPLE VIT!

            vit = SDViT(
                seq_len=input_dim,
                patch_size=patch_size,
                num_classes=self.n_classes,
                channels=input_channels,
                dim=hidden_dim,
                dim_head=dim_head,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                emb_dropout=dropout
            )

            self.encoder = vit
        else:
            assert False, f"Unknown {backbone=}"

        self.softmax = nn.Softmax(dim=1)  # TODO => beware that here we put softmax, we might want to use sigmoid for some applications?

        if self.mode == "quant":
            self.loss = RMSELoss()
        if self.mode == "qual":
            self.loss = nn.CrossEntropyLoss()

        self.y_true = []
        self.y_pred = []

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "RMSProp":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        else:
            assert False, f"Unknown {self.hparams.optimizer=}"
        if self.hparams.lr_scheduler == "reduceonplateau":
            print(
                f"Setting optimizer to reduceonplateau with params {self.hparams.lr_reduceonplateau_factor=}, {self.hparams.lr_reduceonplateau_patience=}, {self.hparams.lr_reduceonplateau_threshold=}, {self.hparams.lr_reduceonplateau_minlr=}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_reduceonplateau_factor,
                                                                   patience=self.hparams.lr_reduceonplateau_patience,
                                                                   threshold=self.hparams.lr_reduceonplateau_threshold,
                                                                   min_lr=self.hparams.lr_reduceonplateau_minlr, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        elif self.hparams.lr_scheduler == "multistep":
            print(f"Setting optimizer to multistep with params {self.hparams.lr_multistep_milestones=}, {self.hparams.lr_multistep_gamma=}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_multistep_milestones, gamma=self.hparams.lr_multistep_gamma,
                                                             verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.hparams.lr_scheduler != "none":
            assert False, f"Unknown {self.hparams.lr_scheduler=}"
        return optimizer

    def forward(self, batch):
        if type(batch) in (tuple, list):
            x, y = batch
        else:
            x = batch
        if self.backbone in ("resnet1d", "net1d"):
            x = x.view(x.size(0), x.size(1), x.size(3))
            x = self.encoder(x)
        elif self.backbone == "homemade":
            x = self.pred(self.fc(self.encoder(x)))
        elif self.backbone == "dense":
            x = self.pred(self.encoder(x))
        elif self.backbone in ("vit_1d", "simple_vit_1d", "smalldataset_vit_1d"):
            x = torch.squeeze(x)  # drop H in H x W
            x = self.encoder(x)
        if self.mode == "qual":
            x = self.softmax(x)  # TODO => beware that here we put softmax, we might want to use sigmoid for some applications?
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        yhat = None
        if self.backbone in ("resnet1d", "net1d"):
            x = x.view(x.size(0), x.size(1), x.size(3))
            yhat = self.encoder(x)
        elif self.backbone == "homemade":
            yhat = self.pred(self.fc(self.encoder(x)))
        elif self.backbone == "dense":
            yhat = self.pred(self.encoder(x))
        elif self.backbone in ("vit_1d", "simple_vit_1d", "smalldataset_vit_1d"):
            x = torch.squeeze(x)  # drop H in H x W
            yhat = self.encoder(x)

        loss = self.loss(yhat, y)

        self.log_dict({
            'loss': loss,
        })

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # encode x to get the mu and variance parameters
        if self.backbone in ("resnet1d", "net1d"):
            x = x.view(x.size(0), x.size(1), x.size(3))
            yhat = self.encoder(x)
        elif self.backbone == "homemade":
            yhat = self.pred(self.fc(self.encoder(x)))
        elif self.backbone == "dense":
            yhat = self.pred(self.encoder(x))
        elif self.backbone in ("vit_1d", "simple_vit_1d", "smalldataset_vit_1d"):
            x = torch.squeeze(x)  # drop H in H x W
            yhat = self.encoder(x)
        else:
            assert False, f"Unknown {self.backbone=}"

        self.y_true.append(y)
        self.y_pred.append(yhat)

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        if self.mode == "quant":

            loss = self.loss(y_pred, y_true)

            # compute mean (Pearson's) correlation coefficient
            if y_true.shape[-1] == 3:
                corrcoefs = torch.corrcoef(torch.transpose(torch.cat([y_true, y_pred], axis=1), 0, 1))
                corrcoefs = torch.diagonal(corrcoefs[0:3, 3::1])
                meancorrcoef = torch.mean(corrcoefs)

                g_loss = self.loss(y_pred[:, [0]], y_true[:, [0]])
                a_loss = self.loss(y_pred[:, [1]], y_true[:, [1]])
                m_loss = self.loss(y_pred[:, [2]], y_true[:, [2]])

                self.log_dict({
                    'val_corcoef': meancorrcoef,
                    'val_loss': loss,
                    'val_g_loss': g_loss,
                    'val_a_loss': a_loss,
                    'val_m_loss': m_loss,
                })

            elif y_true.shape[-1] == 1:
                corcoef = torch.corrcoef(torch.transpose(torch.cat([y_true, y_pred], axis=1), 0, 1))
                corcoef = corcoef[1, 0]

                self.log_dict({
                    'val_corcoef': corcoef,
                    'val_loss': loss,
                })

        if self.mode == "qual":
            ce_loss = self.loss(y_pred, y_true)
            accuracy = torch.sum(torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)) / y_true.shape[0]

            log_dict = {'val_loss': ce_loss, 'val_accuracy': accuracy}
            for i in range(self.n_classes):
                preds = torch.argmax(y_pred[torch.argmax(y_true, dim=1) == i], dim=1)
                log_dict[f"val_accuracy_class_{i}"] = torch.sum(preds == i) / preds.shape[0]

            self.log_dict(log_dict)

        self.y_true.clear()
        self.y_pred.clear()


# %%

if __name__ == "__main__":
    import re

    input_dim = 2560
    input_channels = 5
    encoder_blocks_layout = [
        [65, 65],
        [130, 130],
        [260, 260],
        [520, 520],
        [520, 520],
        [1040, 1040],
        [1040, 1040],
    ]
    maxpool = 2
    kernel_size = 5
    dropout = 0
    stride = 1
    first_layer_groups = 5
    groups = 5
    flatten = "flatten"
    latent_dim = 512
    batchnorm = True
    hidden_activation = "relu"
    output_activation = "none"

    encoder_modules, enc_out_dim = _get_homemade_encoder_v1_base_modules(input_dim=input_dim, input_channels=input_channels, maxpool=maxpool,
                                                                         encoder_blocks_layout=encoder_blocks_layout, kernel_size=kernel_size, stride=stride,
                                                                         dropout=dropout, groups=groups, first_layer_groups=first_layer_groups, flatten=flatten,
                                                                         batchnorm=batchnorm, hidden_activation=hidden_activation)
    encoder = nn.Sequential(*encoder_modules)
    print(f"{encoder}")

    fake_x = torch.randn(size=(32, 5, 1, 2560))
    fake_encoder_output = encoder(fake_x)
    print(f"{fake_encoder_output.shape=}")

    # print(f"{enc_lastconv_channels=}")
    # print(f"{enc_lastconv_dim=}")
    print(f"{enc_out_dim=}")

    decoder_modules = _get_homemade_decoder_v1_base_modules(input_dim=input_dim, latent_dim=latent_dim, include_initial_dense=False,
                                                            input_channels=input_channels,
                                                            maxpool=maxpool,
                                                            encoder_blocks_layout=encoder_blocks_layout,
                                                            kernel_size=kernel_size,
                                                            stride=stride, first_layer_groups=first_layer_groups,
                                                            groups=groups, batchnorm=batchnorm, hidden_activation=hidden_activation,
                                                            output_activation=output_activation)

    decoder = nn.Sequential(*decoder_modules)
    print(f"{decoder=}")

    fake_latent = torch.randn(size=(32, 512))
    fake_decoder_output = decoder(fake_latent)
    print(f"{fake_decoder_output.shape=}")

    fake_hidden = fake_latent
    print(f"layer=-1 {'(input)'.rjust(17)}:  input_shape={fake_hidden.shape}")
    for i in range(len(decoder_modules)):
        ldtype = re.sub("^.+ of ([^(]+)[(].+$", "\\1", str(decoder_modules[i].type))
        ldtype = f"({ldtype})"
        fake_hidden = decoder_modules[i](fake_hidden)
        print(f"layer={i:02d} {ldtype.rjust(17)}: output_shape={fake_hidden.shape}")
