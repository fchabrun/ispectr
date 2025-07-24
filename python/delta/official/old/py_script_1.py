# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:41:53 2019

@author: Floris Chabrun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load raw data, exported in R
# see report for further details on the structure of this .csv file
path_to_csv_input_file = "./CSV_data_2.csv"

raw = pd.read_csv(path_to_csv_input_file)

# here we define the size of a curve -> 304 points
initial_spe_width=298
specifications_spe_width=304

# we define the names of the columns for extracting data from the .csv file
curve_columns = [c for c in raw.columns if c[:4]=='ELPx'] # names: ELPx1 to ELPx304
Kcurve_columns = [c for c in raw.columns if c[:2]=='Kx'] # names: Kx1 to Kx304
Lcurve_columns = [c for c in raw.columns if c[:2]=='Lx'] # names: Lx1 to Lx304
Gcurve_columns = [c for c in raw.columns if c[:4]=='IgGx'] # names: IgGx1 to IgGx304
Acurve_columns = [c for c in raw.columns if c[:4]=='IgAx'] # names: IgAx1 to IgAx304
Mcurve_columns = [c for c in raw.columns if c[:4]=='IgMx'] # names: IgMx1 to IgMx304

assert len(curve_columns) == initial_spe_width, "Reference curves are not 304-points long"
assert len(Gcurve_columns) == initial_spe_width, "Heavy chain G curves are not 304-points long"
assert len(Acurve_columns) == initial_spe_width, "Heavy chain A curves are not 304-points long"
assert len(Mcurve_columns) == initial_spe_width, "Heavy chain M curves are not 304-points long"
assert len(Kcurve_columns) == initial_spe_width, "Light chain kappa curves are not 304-points long"
assert len(Lcurve_columns) == initial_spe_width, "Light chain lambda curves are not 304-points long"

# convert raw data to numpy
x=raw.loc[:,curve_columns].to_numpy()
xK=raw.loc[:,Kcurve_columns].to_numpy()
xL=raw.loc[:,Lcurve_columns].to_numpy()
xG=raw.loc[:,Gcurve_columns].to_numpy()
xA=raw.loc[:,Acurve_columns].to_numpy()
xM=raw.loc[:,Mcurve_columns].to_numpy()

# add padding, i.e. zeros before and after the curve in order
# to convert 298-points long curves into 304-points long curves
def addPaddingToArray(arr, initial_spe_width = initial_spe_width, specifications_spe_width = specifications_spe_width):
    padding_before = (specifications_spe_width-initial_spe_width) // 2
    padding_after = (specifications_spe_width-initial_spe_width) - padding_before
    return np.concatenate([np.zeros((arr.shape[0],padding_before)),
                           arr,
                           np.zeros((arr.shape[0],padding_after))], axis=-1)

x,xK,xL,xG,xA,xM = [addPaddingToArray(unpadded_array) for unpadded_array in (x,xK,xL,xG,xA,xM)]

# normalize between 0 and 1
x = x/(np.max(x, axis = 1)[:,None])
xK = xK/(np.max(xK, axis = 1)[:,None])
xL = xL/(np.max(xL, axis = 1)[:,None])
xG = xG/(np.max(xG, axis = 1)[:,None])
xA = xA/(np.max(xA, axis = 1)[:,None])
xM = xM/(np.max(xM, axis = 1)[:,None])

# %%

# prepare annotation array -> empty for now (only zeros)
extended_class_maps = np.zeros(x.shape + (5,))

# %%

spe_width = specifications_spe_width

# we define our own function for plotting curves with m-spikes
# this function will be used for annotating curves
# as well as checking that curves are correct
def plotExtendedIFMaps(ix):
    class_map=extended_class_maps[ix].max(axis=-1)
    curve_values = x[ix,:]
    x_max = np.argmax(curve_values)
    plt.figure(figsize=(18,9))
    plt.subplot(6,2,1)
    plt.text(spe_width-1,1,"Curve {}".format(ix), horizontalalignment="right", verticalalignment='top')
    # plt.plot(np.arange(0,spe_width), (y_annotations_unchecked[ix,:]), '-', color='blue')
    plt.plot(np.arange(0,spe_width), curve_values, '-', color = 'black')
    plt.text(0,1, 'Reference', verticalalignment='top', weight='bold')
    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=curve_values[peak_start:peak_end], color="red")
        # plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')
    plt.text(x_max+5,.5,x_max, verticalalignment='top', color='red')
        
    plt.subplot(6,2,2)
    plt.text(spe_width-1,1,"Curve {} (deltas)".format(ix), horizontalalignment="right", verticalalignment='top')
    # plt.plot(np.arange(0,spe_width), (y_annotations_unchecked[ix,:]), '-', color='blue')
    plt.plot(np.arange(0,spe_width), curve_values, '-', color = 'black')
    plt.text(0,1, 'Reference', verticalalignment='top', weight='bold')
    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=curve_values[peak_start:peak_end], color="red")
        # plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')
    plt.text(x_max+5,.5,x_max, verticalalignment='top', color='red')

    annot_ig_display = ['G-CURVE','A-CURVE','M-CURVE','K-CURVE','L-CURVE']
    for ig_plane,ig_values in enumerate([xG[ix],xA[ix],xM[ix],xK[ix],xL[ix]]):
        x_max = np.argmax(ig_values)
        plt.subplot(6,2,3+ig_plane*2)
        plt.plot(np.arange(0,spe_width), ig_values, '-', color = 'black')
        
        for peak_dim in np.concatenate([np.setdiff1d(np.arange(0,5),ig_plane), [ig_plane,]]):
            for peak_start, peak_end in zip(np.where(np.diff(extended_class_maps[ix,:,peak_dim])==1)[0]+1, np.where(np.diff(extended_class_maps[ix,:,peak_dim])==-1)[0]+1):
                if peak_dim==ig_plane:
                    color='red'
                else:
                    color='blue'
                plt.plot(np.arange(peak_start,peak_end), ig_values[peak_start:peak_end], '-', color = color)
        plt.text(0,1,annot_ig_display[ig_plane], verticalalignment='top', weight='bold')
        plt.text(x_max+2,1,x_max, verticalalignment='top', color='red')
        
        ig_values = curve_values-ig_values
        plt.subplot(6,2,4+ig_plane*2)
        plt.plot(np.arange(0,spe_width), ig_values, '-', color = 'black')
        for peak_dim in np.concatenate([np.setdiff1d(np.arange(0,5),ig_plane), [ig_plane,]]):
            for peak_start, peak_end in zip(np.where(np.diff(extended_class_maps[ix,:,peak_dim])==1)[0]+1, np.where(np.diff(extended_class_maps[ix,:,peak_dim])==-1)[0]+1):
                if peak_dim==ig_plane:
                    color='red'
                else:
                    color='blue'
                plt.plot(np.arange(peak_start,peak_end), ig_values[peak_start:peak_end], '-', color = color)
        plt.ylim(-1,1)
        plt.text(0,.9,annot_ig_display[ig_plane], verticalalignment='top')
        plt.text(x_max+2,1,x_max, verticalalignment='top', color='red')
        plt.tight_layout()

# %%

# we define constants in order to speed up the workflow
HEAVY_G = 0
HEAVY_A = 1
HEAVY_M = 2
LIGHT_K = -2
LIGHT_L = -1

# %%

# for each sample, we manually enter the coordinates at which there are m-spikes
# as well as the nature of the m-spike (G, A, M, kappa, lambda)

curve_id = 0 # for sample 0
plotExtendedIFMaps(curve_id) # plot the curves: we see 1 M-spike : IgGlambda between 243 and 251
extended_class_maps[curve_id,...] = 0 # raz previous changes
extended_class_maps[curve_id,243:251,[HEAVY_G,LIGHT_L]] = 1 # set values between 243 (included) and 251 (excluded) to 1 for G and L dimensions -> meaning there's a IgGl m-spike at this location
plotExtendedIFMaps(curve_id) # visually check result

# ...

curve_id = -1 # last sample
plotExtendedIFMaps(curve_id) # plot the curves: we see  # 1 M-spike : IgMkappa between 223 and 233
extended_class_maps[curve_id,...] = 0 # raz previous changes
extended_class_maps[curve_id,223:233,[HEAVY_M,LIGHT_K]] = 1 # set values between 223 (included) and 233 (excluded) to 1 for M and K dimensions -> meaning there's a IgMk m-spike at this location
plotExtendedIFMaps(curve_id) # visually check result

# %%

# Finally, convert our data to numpy arrays which will be used for training and testing models
if_x = np.stack([x,xG,xA,xM,xK,xL], axis=-1)
if_y = extended_class_maps

# and save them as numpy array files
path_to_output_x = "./NPY_file_X.npy"
path_to_output_y = "./NPY_file_Y.npy"

np.save(path_to_output_x, if_x)
np.save(path_to_output_y, if_y)