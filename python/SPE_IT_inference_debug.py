"""#############################################################################
#################### SPE IT MODEL TRAINING #####################################
author : Chabrun Floris and Dieu Xavier
date : 22/11/2024
Training segmentation models for immunosubtraction data
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

df = pd.read_csv(r"C:\Users\afors\Documents\Projects\SPE_IT\data_sa\sa_is_full.csv")

spe_range = np.arange(305)
spe_range = spe_range[1:]
spe_range_str = list(map(str, spe_range))

sa_elp = df.loc[ : , ['ELP_x'+i for i in spe_range_str]]
sa_g = df.loc[ : , ['IgG_x'+i for i in spe_range_str]]
sa_a = df.loc[ : , ['IgA_x'+i for i in spe_range_str]]
sa_m = df.loc[ : , ['IgM_x'+i for i in spe_range_str]]
sa_k = df.loc[ : , ['K_x'+i for i in spe_range_str]]
sa_l = df.loc[ : , ['L_x'+i for i in spe_range_str]]

sa_elp = sa_elp.to_numpy()
sa_g = sa_g.to_numpy()
sa_a = sa_a.to_numpy()
sa_m = sa_m.to_numpy()
sa_k = sa_k.to_numpy()
sa_l = sa_l.to_numpy()

# we need to create a dataset with shape (batch, spe_length, tracks) i.e. (719,304,6)
x_test = np.dstack([sa_elp, sa_g, sa_a, sa_m, sa_k, sa_l])

# x_test = np.moveaxis(np.expand_dims(x_test, 2), (0,1,2,3), (0,3,2,1))


sa_dataset = ISDataset(if_x=x_test, if_y=np.zeros((719,304,5)), smoothing=False, normalize=False, coarse_dropout=False, permute=None)


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
model = pl_IS_model.load_from_checkpoint(r"C:\Users\afors\Documents\Projects\SPE_IT\Best_model\mednext_L\last.ckpt")

# predict on validation data
sa_outputs = trainer.predict(model, dataloaders=test_loader)
# note: in pytorch, the output is a list of N elements, N being the number of batches => so we have to convert that to a np array
# new: make sure we are not using logits but probabilities
sa_preds = torch.nn.Sigmoid()(torch.cat(sa_outputs)).detach().cpu().numpy()

sa_preds.shape
sa_preds[0,4,:].max() # Ml for the fisrt trace


def plot_IT_predictions(idx, sample_x, sample_pred, debug):
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


plot_IT_predictions(0, x_test[0, ...], sa_preds[0,...], debug='inline')
plot_IT_predictions(1, x_test[1, ...], sa_preds[1,...], debug='inline')
plot_IT_predictions(2, x_test[2, ...], sa_preds[2,...], debug='inline')
plot_IT_predictions(3, x_test[3, ...], sa_preds[3,...], debug='inline')
plot_IT_predictions(4, x_test[4, ...], sa_preds[4,...], debug='inline')
