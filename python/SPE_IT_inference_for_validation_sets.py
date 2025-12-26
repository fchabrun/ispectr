"""#############################################################################
#################### SPE IT MODEL TRAINING #####################################
author : Chabrun Floris and Dieu Xavier
date : 22/11/2024
This script is used to perform inference (i.e. export validation predictions) for validation sets (not approached in the training script)
# Those predictions can then be used in the jupyter notebook scripts
#############################################################################"""

# general modules
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# PYTORCH
import torch
import torch.utils.data as data
import lightning as pl

# custom modules
from spep_assets.spep_data import ISDataset
from spep_assets.spep_models import pl_IS_model

# loading SA data

TEST_DATASET_NAME = "sa_2025"
TEST_DATA_PATH = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\final_datasets\capetown\2025_12_24\full_dataset.csv"
MODEL_PATH = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\output\mednext_L_TEST2025"

raw_data = pd.read_csv(TEST_DATA_PATH)
# new: filter out exclude
raw_data = raw_data[~raw_data.exclude].copy()

X_test = np.stack([raw_data.loc[:, [f'{trace}_{x+1}' for x in np.arange(304)]].values for trace in ['ELP', 'IgG', 'IgA', 'IgM', 'K', 'L']],
                  axis=-1)
y_test = np.stack([raw_data.loc[:, [f'segm_{trace}_{x+1}' for x in np.arange(304)]].values for trace in ['IgG', 'IgA', 'IgM', 'K', 'L']],
                  axis=-1)
print(f"Loaded X test data of shape {X_test.shape=}")
print(f"Loaded X test data of shape {y_test.shape=}")

sa_dataset = ISDataset(if_x=X_test, if_y=np.zeros((len(X_test), 304, 5)), smoothing=False, normalize=False, coarse_dropout=False, permute=None)

num_workers = 0  # how many processes will load data in parallel; 0 for none

# create our dataset loader for train data
test_loader = data.DataLoader(
    sa_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False
    # if we set >1 loader, we want them to be persistent, i.e. not being instantiated again between each epoch
)

trainer = pl.Trainer()

# loading model from save
model = pl_IS_model.load_from_checkpoint(os.path.join(MODEL_PATH, "last.ckpt"))

# predict on validation data
test_outputs = trainer.predict(model, dataloaders=test_loader)
# note: in pytorch, the output is a list of N elements, N being the number of batches => so we have to convert that to a np array
# new: make sure we are not using logits but probabilities
test_preds = torch.nn.Sigmoid()(torch.cat(test_outputs)).detach().cpu().numpy()


def plot_IT_predictions(idx, sample_x, sample_y, sample_pred, debug):
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    # on récupère la class map (binarisée)
    # class_map = y[ix].max(axis=1)
    curve_values = sample_x
    for num, col, lab in zip(range(6), ['black', 'purple', 'pink', 'green', 'red', 'blue'],
                             ['Ref', 'G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304), curve_values[:, num], '-', color=col, label=lab)
    plt.title('Valid set curve index {}'.format(idx))

    plt.legend()
    # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
    #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

    plt.subplot(3, 1, 2)
    for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304) + 1, sample_y[:, num] / 5 + (4 - num) / 5, '-', color=col, label=lab)
    plt.ylim(-.05, 1.05)
    plt.legend()
    plt.title('Ground truth maps')

    plt.subplot(3, 1, 3)
    for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304) + 1, sample_pred[num, :] / 5 + (4 - num) / 5, '-', color=col, label=lab)
    plt.ylim(-.05, 1.05)
    plt.legend()
    plt.title('Predicted maps')

    if debug != "inline":
        plt.savefig(os.path.join(debug, f"pred_plot_{idx=}.png"))
        plt.close()
    else:
        plt.show()


idx = 0
for idx in range(5):
    plot_IT_predictions(idx,
                        sample_x=X_test[idx, ...],
                        sample_y=y_test[idx, ...],
                        sample_pred=test_preds[idx, ...],
                        debug='inline')

np.save(os.path.join(MODEL_PATH, f"{TEST_DATASET_NAME}_preds.npy"), test_preds)
np.save(os.path.join(MODEL_PATH, f"{TEST_DATASET_NAME}_truth.npy"), y_test)
np.save(os.path.join(MODEL_PATH, f"{TEST_DATASET_NAME}_x.npy"), X_test)
