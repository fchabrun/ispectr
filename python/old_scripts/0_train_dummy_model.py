"""
Created Nov 20 by Chabrun F
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from old_assets.spep_dl import SupervisedModule
from spep_assets.spep_data import ISDataset

import torch
import torch.utils.data as data
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from spep_assets.spep_figures import pp_size, plot_roc
from spep_assets.spep_stats import get_bootstrap_metric_ci

# where to write everything
output_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\output"

# load lemans data

# load x array
# data is already normalized between 0-1 and zero-padded to a 304 width
if_x = np.load(r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\data\lemans_2018\already_proc\if_v1_x.npy")

# load y array
if_y = np.load(r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\data\lemans_2018\already_proc\if_v1_y.npy")

# note: we should create .h5 files to easily load the data using a data manager if we have a lot of samples!

debug_plots = True

if debug_plots:
    # show the firts sample of the dataset
    is_tracks = ["ELP", "IgG", "IgA", "IgM", "K", "L"]

    i = 1
    plt.figure(figsize=(12, 12))
    for j in range(6):
        plt.subplot(6, 2, j * 2 + 1)
        sns.lineplot(x=np.arange(304), y=if_x[i, :, j])  # plot data
        if j > 0:  # plot annotation map
            plt.subplot(6, 2, j * 2 + 2)
            sns.lineplot(x=np.arange(304), y=if_y[i, :, j - 1])
    plt.tight_layout()
    plt.show()

# %%

# for the "dummy" task, we'll try to train a model to predict : IgG peak vs no IgG peak
# so we'll just look at the peak segmentation maps
# and detect whether there is an IgG peak
if_y_igg_binary = if_y[..., 0].max(axis=1)  # now we have a vector of shape N, so 1 value per sample, which is 0 if no IgG peak or 1 if any IgG peak
# reshape to 2 output neurons (1 => 01, 0 => 10)
if_y_igg_binary = np.stack([1 - if_y_igg_binary, if_y_igg_binary], axis=1)

# partition
if_x_train, if_x_test, if_y_igg_binary_train, if_y_igg_binary_test = train_test_split(if_x, if_y_igg_binary,
                                                                                      test_size=.2,
                                                                                      random_state=1, shuffle=True,
                                                                                      stratify=if_y_igg_binary)

train_dataset = ISDataset(if_x=if_x_train, if_y=if_y_igg_binary_train, smoothing=False, normalize=False, coarse_dropout=False)
test_dataset = ISDataset(if_x=if_x_test, if_y=if_y_igg_binary_test, smoothing=False, normalize=False, coarse_dropout=False)

num_workers = 8  # how many processes will load data in parallel; 0 for none

# train create our dataset loaders
train_loader = data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False
    # if we set >1 loader, we want them to be persistent, i.e. not being instanciated again between each epoch
)

validation_loader = data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False
)

if debug_plots:
    # just to check if the loader works OK
    first_train_batch = next(iter(train_loader))
    x, y = first_train_batch
    print(f"First training batch: {x.shape=} / {y.shape=}")

# %%

# TODO pré-entraîner modèle sur une autre tâche
# TODO masquer les pics pour forcer le modèle à regarder autour de l'albumine (expliquer à Xavier!)

# TODO note => SupervisedModule works OK
# TODO other models in spep_dl => they worked at some point, but a hell lot of modifications were made in between so really not so sure right now

# TODO instead of using grouped convolutions with 1D model, we may want to try using regular convolutions with a 2D model? (1 channel, but height = 6)
# TODO limit with this solution => 2nd dimension will move over "channels" (IgG/A/M/k/l) always in the same order so consider that "some IgG pattern" close to some "IgA pattern" and not "IgG" right before "Kappa" for instance... so maybe not the best solution?

# TODO segmentation instead of classification

# instanciate our first model
model = SupervisedModule(n_classes=2,  # 2 output neurons => igg vs no igg
                         mode="qual",  # qualitative prediction
                         input_dim=304,
                         input_channels=6,
                         latent_dim=128,
                         kernel_size=5,
                         flatten="flatten",
                         stride=1,
                         dropout=.0,
                         groups=6,
                         first_layer_groups=6,
                         maxpool=2,
                         encoder_blocks_layout=[[60, 60],
                                                [120, 120],
                                                [120, 120],
                                                [240, 240],
                                                [240, 240],
                                                [480, 480],
                                                [480, 480],
                                                [960, 960],
                                                [960, 960],
                                                ],
                         backbone="homemade",
                         batchnorm=True,
                         hidden_activation="selu",
                         optimizer="Adam",
                         lr=1e-4, lr_scheduler="reduceonplateau",
                         lr_reduceonplateau_factor=.5, lr_reduceonplateau_patience=3, lr_reduceonplateau_threshold=1e-2, lr_reduceonplateau_minlr=1e-6,
                         lr_multistep_milestones=None, lr_multistep_gamma=None,
                         )

# create our trainer that will handle training
logger = CSVLogger(save_dir=output_path, name="logs")
callbacks = [ModelCheckpoint(dirpath=output_path, save_weights_only=True,
                             mode="min", monitor="val_loss",
                             save_last=True),
             EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"),
             ]

trainer_args = {'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                'devices': 'auto',
                'num_nodes': 1,
                'strategy': 'auto'}

trainer = pl.Trainer(
    default_root_dir=output_path,
    **trainer_args,
    max_epochs=100,
    log_every_n_steps=100,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=logger,
)

# fit model
trainer.fit(model, train_loader, validation_loader)

# %% validation

# if we specified a model checkpoint => reload this specific checkpoint
# model = SupervisedModule.load_from_checkpoint(reload_checkpoint)

validation_outputs = trainer.predict(model, dataloaders=validation_loader)
# note: in pytorch, the output is a list of N elements, N being the number of batches => so we have to convert that to a np array
validation_preds = torch.cat(validation_outputs).detach().cpu().numpy()

# do a nice plot

plot_data = pd.DataFrame(dict(ground_truth=if_y_igg_binary_test[:, 1],
                              prediction=validation_preds[:, 1]))

plt.figure(figsize=(pp_size, pp_size))
plot_roc(y=plot_data["ground_truth"].values.astype(int), y_=plot_data["prediction"], confidence_level=.99)
# plt.savefig(os.path.join(args.output_path, FIGURES_PATH, "roc_predict_race_from_batch_order.png"))
# plt.close()
plt.show()

# ROC-AUC = 0.94 [0.90-0.97]
from sklearn.metrics import f1_score

f1_score(y_true=plot_data["ground_truth"].values.astype(int), y_pred=(plot_data["prediction"] > .5) * 1)  # 0.9134199

get_bootstrap_metric_ci(groundtruths=plot_data["ground_truth"].values.astype(int),
                        preds=(plot_data["prediction"] > .5) * 1,
                        metric="f1",
                        bootstraps=1000, alpha=.01)  # 0.91 [0.88-0.94] >>> Olivier Bec-De-Lièvre
