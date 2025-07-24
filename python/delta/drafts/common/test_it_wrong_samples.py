# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# Pour entraîner un modèle :
# - Modifier la valeur des variables au début de la première section (BASE PART) :
#       paths pour l'input et l'output
# - Exécuter la première section (BASE PART)
# - Modifier la valeur des variables au début de la 2e section (TRAINING PART) :
#       hyperparamètres utilisés pour l'entraînement
# - Exécuter la deuxième section (TRAINING PART)

# Pour tester un modèle :
# - Modifier la valeur des variables au début de la première section (BASE PART) :
#       paths pour l'input et l'output
# Exécuter la première section (BASE PART)
# - Modifier la valeur des variables au début de la 2e section (INFERENCE PART) :
#       en fonction de l'input/output du modèle utilisé et du type de post-processing nécessaire
# Exécuter la troisième section (INFERENCE PART)

#################################
# HYPERPARAMETERS AND CONSTANTS #
#################################

# Xavier
# since the callbacks will handle adapting the learning rate and stopping training when no performance improvement is observed after a few epochs
# PATH_IN = "D:/Anaconda datasets/Capillarys/IF_transformer" # path for data (in)
# PATH_OUT = "D:/Anaconda datasets/Capillarys/IF_transformer/output" # path for output (out)
normc = False
# Floris
PATH_IN = 'C:/Users/admin/Documents/Capillarys/data/2021/ifs/sebia'
PATH_OUT = 'C:/Users/admin/Documents/Capillarys/temp2021'

# %%

##########################################################
##########################################################
#####                   BASE PART                    #####
##########################################################
##########################################################

# please make sure you have the library "Xlsxwriter" installed for exporting the "preddy" contingency table

import numpy as np
import pandas as pd
import os
from colour import Color
from tqdm import tqdm, trange
import tensorflow as tf

##############################
# LOAD AND PREPROCESS DATA ###
##############################

# LOAD DATA
if_x = np.load(os.path.join(PATH_IN,'X_wrong_samples.npy'))
if_y = np.load(os.path.join(PATH_IN,'y_wrong_samples_real.npy'))

# PARTITIONING
# we should have 3 datasets:
# 1) training -> data used for training models
# 2) validation -> data used for monitoring loss and metrics during training, stop training when performance doesn't improve and compare architectures when changing hyperparameters
# 3) test -> for testing the performance of the final model on new, unseen data

# for this dummy example, just copy arrays for training, validation and test
# if using the real dataset, partition first

x_test = if_x
y_test = if_y

spe_width = 304 # define the size of the input (i.e. 304 points)

# %%

##########################################################
##########################################################
#####                 INFERENCE PART                 #####
##########################################################
##########################################################

INPUT_DIMENSIONS = 2
POST_PROCESS_METHOD = 'invert_predictions'
POST_PROCESS_SQUARE = True
POST_PROCESS_SMOOTH = False

if INPUT_DIMENSIONS == 2: # linearize dataset
    # prepare input for model
    # initially we have an array of dim (N_SAMPLES x 304 x 6)
    # i.e. 304 point-curves and 6 dimensions: ref, G, A, M, k, l
    # our model is only capable of analyzing curves "one by one"
    # so we have to reshape our array with new dim: (N_SAMPLES*5, 304)
    test_input = np.stack([x_test[i,:,d] for i in range(x_test.shape[0]) for d in range(1,6)], axis=0)
    # in addition, we had 2 new "empty" dimensions
    # the first new dimension is because we perform 2D-convolutions
    # so our curve is considered as an image of width x height = 304 x 1
    # the second new dimension is for stacking the output of all convolution filters performed by every convolutional layer
    # i.e. a 304x1x1 curve analyzed by a 32-filter conv layer will output a 304x1x32 array
    test_input = np.expand_dims(test_input, axis=(2,3))
elif INPUT_DIMENSIONS == 3: # keep original dimensions of the dataset
    test_input = x_test
elif INPUT_DIMENSIONS == 4: # flatten then stack ref with GAMkl
    test_input = np.concatenate([np.expand_dims(np.tile(x_test[...,[0]],5),-1), np.expand_dims(x_test[...,1:],-1)], -1).copy()
    
model = tf.keras.models.load_model(os.path.join(PATH_OUT, "segifaug-parallel_unetv1wide-loss-binary_crossentropy.h5"), compile=False)

y_test_raw_ = model.predict(test_input)

##################################
#  POST PROCESS MODEL'S OUTPUT   #    
##################################

size = x_test.shape[0]

if INPUT_DIMENSIONS==2:
    # post-process neural network output
    # first we simply discard the last dimensions (only needed by tensorflow for stacking conv filters)
    y_test_ = y_test_raw_[...,0,0]
    
    # we have to convert the dimensions of the predicted array
    # back from : (N_SAMPLES*5 x 304)
    # to : (N_SAMPLES x 304 x 5)
    y_test_ = np.transpose(np.stack([y_test_[[b*5+i for i in range(5)],...] for b in range(size)], axis=-1), axes=(2,1,0))
elif INPUT_DIMENSIONS==3:
    y_test_ = y_test_raw_.copy() # keep original dimension for output
elif INPUT_DIMENSIONS==4:
    y_test_raw_ = y_test_raw_[:,:,:,0]
    y_test_ = y_test_raw_.copy()

if POST_PROCESS_METHOD == 'invert_predictions':
    # finally, invert predictions
    # e.g. if a m-spike is detected on all curves except G and k, then it's a Gk mspike
    # so all curves should be only zeros,
    # except the G and k which should have ones at the position of the mspike
    
    # for each sample (i.e. group of 5 G, A, M, k, l curves)
    # we compute the "reference" curve
    # i.e. the curve with max predicted value (along all 5 axes) for each of the 304 points
    y_test_ref_ = np.max(y_test_, axis=-1)
    # then we invert all predicted curves and multiply them by the reference curve
    # which will give for instance :
    # value for G = 1 (mspike detected in G), value for ref = 1 (mspike detected at this location) -> (1-1)*1 = 0 (the mspike IS NOT a G mspike)
    # value for G = 0 (NO mspike detected in G), value for ref = 1 (mspike detected at this location) -> (1-0)*1 = 1 (the mspike IS a G mspike)
    
    for dim in range(5):
        if POST_PROCESS_SQUARE:
            y_test_[...,dim] = np.power((1-y_test_[...,dim]), 1)*np.power(y_test_ref_, 2) # en mettant les courbes au carré, on ne modifie pas les valeurs proches de 1 mais on diminue exponentiellement les valeurs faibles, permettant de réduire l'effet de ghosting
        else:
            y_test_[...,dim] = (1-y_test_[...,dim])*y_test_ref_
        
    if POST_PROCESS_SMOOTH:
        def smooth_curve(x, window_len=3):
            new_values = []
            for j in range(len(x)):
                window_start = max(0,j-(window_len-1)//2)
                window_end = min(len(x)-1,j+(window_len-1)//2)+1
                new_val = np.mean(x[window_start:window_end])
                new_values.append(new_val)
            return np.array(new_values)
        for i in tqdm(range(y_test_.shape[0])):
            for d in range(y_test_.shape[-1]):
                y_test_[i,...,d] = smooth_curve(y_test_[i,...,d])
                
##################################
#     PLOTS (FOR DEBUGGING)      #    
##################################

# Simple plot function : raw, but efficient
def debugPlotSample(ix=0):
    from matplotlib import pyplot as plt
    """
    debug function: plot the predictions (raw and processed) for sample with index = ix
    """
    fig = plt.figure(figsize=(16,8)) # open the figure
    ax_dict = fig.subplot_mosaic([["Ref","Ref","Ref", "Ref"],
                                  ["G - {}".format(s) for s in ("input" ,"ground truth", "raw preds", "proc preds")],
                                  ["A - {}".format(s) for s in ("input" ,"ground truth", "raw preds", "proc preds")],
                                  ["M - {}".format(s) for s in ("input" ,"ground truth", "raw preds", "proc preds")],
                                  ["k - {}".format(s) for s in ("input" ,"ground truth", "raw preds", "proc preds")],
                                  ["l - {}".format(s) for s in ("input" ,"ground truth", "raw preds", "proc preds")]])
    # plot reference curve
    ax_dict['Ref'].plot(np.arange(304), x_test[ix,:,0])
    ax_dict['Ref'].set_ylim((0,1))
    ax_dict['Ref'].text(0,1,"Reference curve", verticalalignment="top")
    # plot input curve, ground truth annotations, raw predictions and post-processed predictions
    # in respectively 1st, 2nd, 3rd and 4th columns
    
    for dim_name, d in zip(("G","A","M","k","l"),range(5)): # for each of the 6 curves: ref, G, A, M, k, l
        # first column : plot the input curve
        ax_dict["{} - input".format(dim_name)].plot(np.arange(304), x_test[ix,:,d+1])
        ax_dict["{} - input".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - input".format(dim_name)].text(0,1,"{} - input curve".format(dim_name), verticalalignment="top")
        # second column : plot the gt annotations
        # ax_dict["{} - ground truth".format(dim_name)].plot(np.arange(304), y_test[ix,:,d])
        # ax_dict["{} - ground truth".format(dim_name)].set_ylim((0,1))
        # ax_dict["{} - ground truth".format(dim_name)].text(0,1,"{} - ground truth".format(dim_name), verticalalignment="top")
        # third column : plot the raw predictions
        ax_dict["{} - raw preds".format(dim_name)].plot(np.arange(304), y_test_raw_[ix*5+d,:,0,0])
        ax_dict["{} - raw preds".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - raw preds".format(dim_name)].text(0,1,"{} - raw preds".format(dim_name), verticalalignment="top")
        # fourth column : plot the processed predictions
        ax_dict["{} - proc preds".format(dim_name)].plot(np.arange(304), y_test_[ix,:,d])
        ax_dict["{} - proc preds".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - proc preds".format(dim_name)].text(0,1,"{} - proc preds".format(dim_name), verticalalignment="top")
    plt.tight_layout()
    plt.show()
        
debugPlotSample(0) # simple visual test to check that the model outputs consistent predictions
# # and that our post-processing algorithms works OK

debugPlotSample(1)
debugPlotSample(2)
debugPlotSample(3)
debugPlotSample(4)
debugPlotSample(5)
debugPlotSample(6)
debugPlotSample(7)

# More refined function : prettier, but less information displayed
def plotITPredictions(ix):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(14,8))
    plt.subplot(3,1,1)
    # on récupère la class map (binarisée)
    # class_map = y[ix].max(axis=1)
    curve_values = x_test[ix,:]
    for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width), curve_values[:,num], '-', color = col, label = lab)
    plt.title('Test set sample #{}'.format(ix))
    plt.legend()
    # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
    #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

    # on plot aussi les autres courbes
    plt.subplot(3,1,2)
    for num,col,lab in zip(range(5), ['purple','pink','green','red','blue'], ['G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width)+1, y_test[ix,:,num]/5+(4-num)/5, '-', color = col, label = lab)
    plt.ylim(-.05,1.05)
    plt.legend()
    plt.title('Ground truth maps')
    
    plt.subplot(3,1,3)
    for num,col,lab in zip(range(5), ['purple','pink','green','red','blue'], ['G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width)+1, y_test_[ix,:,num]/5+(4-num)/5, '-', color = col, label = lab)
    plt.ylim(-.05,1.05)
    plt.legend()
    plt.title('Predicted maps')

plotITPredictions(0)

##################################
#            METRICS             #
##################################
    
# DEFINE OUR CUSTOM METRIC FUNCTION (Intersection over Union, IoU, Jaccard Index), with tensorflow methods

def curve_iou(y_true, y_pred, smooth = 1e-5):
    trh = tf.cast(tf.greater(y_true, .5), 'double')
    prd = tf.cast(tf.greater(y_pred, .5), 'double')
    i = tf.cast(tf.greater(trh+prd, 1), 'double')
    u = tf.cast(tf.greater(trh+prd, 0), 'double')
    i = tf.reduce_sum(i)
    u = tf.reduce_sum(u)
    return (smooth+i) / (smooth+u)

# we below compute accuracy (pr) and iou (both using our tensorflow method + a regular iou method without tensorflow)
# accuracy (pr) = percentage of correctly classified points
#     i.e. for each point among the N_SAMPLES*304*5 values in the dataset
#     we consider it correctly classified if round(predicted value) == ground truth value (if threshold==.5)
# iou : for each sample S and each dimension D (i.e., curve)
#     sum( prediction(S,D)>.5 AND ground truth(S,D)==1 ) / sum( prediction(S,D)>.5 OR ground truth(S,D)==1 )
threshold = .5
points=np.arange(1,spe_width+1,1)
pr=np.zeros((y_test_.shape[0],5))
iou=np.zeros((y_test_.shape[0],5))
iou_tf=np.zeros((y_test_.shape[0],5))
for ix in trange(y_test_.shape[0]):
    for dim in range(5):
        gt = y_test[ix,:,dim] # get ground truth annotations for this sample & this curve
        pd_ = (y_test_[ix,:,dim]>threshold)*1 # get predictions for this sample & this curve (transform into 0/1 according to selected threshold)
        u = np.sum(gt+pd_>0) # union : either predictions OR ground truth == 1
        i = np.sum(gt+pd_==2) # intersect : both predictions AND ground truth == 1
        if np.isfinite(u):
            iou[ix,dim] = i/u
        else: # u==0 -> division by zero
            iou[ix,dim] = np.nan
        pr[ix,dim] = np.sum(gt==pd_)/spe_width
        # iou_tf[ix,dim] = float(curve_iou(gt.astype('double'),pd_.astype('double')))
        iou_tf[ix,dim] = curve_iou(y_test[ix,:,dim].astype('double'),y_test_[ix,:,dim].astype('double'))

# print results        
print("")

# iou "réelle", i.e. points > ou < au seuil, vs. points à 0 ou 1
print("Mean IoU (real, vs. treshold={}):".format(threshold))
for k in range(iou.shape[1]):
    print("    > Fraction '{}': {:.2f} +- {:.2f}".format(['G','A','M','k','l'][k],np.nanmean(iou[:,k]),np.nanstd(iou[:,k])))

# l'iou calculée par méthodes tensorflow est plus intéressante car elle n'utilise pas de seuil
# mais tient compte uniquement de l'ordre des valeurs (i.e. similaire à une AUC)
print("Mean IoU (tensorflow, threshold-independant):")
for k in range(iou_tf.shape[1]):
    print("    > Fraction '{}': {:.2f} +- {:.2f}".format(['G','A','M','k','l'][k],np.nanmean(iou_tf[:,k]),np.nanstd(iou_tf[:,k])))

##############################
#### TEST PAR ECHANTILLON ####
##############################

# pour chaque échantillon, on va d'abord déterminer une classe "ground truth"
# en fonction des anomalies retrouvées
def classifySample(postprocessed_predicted_curves):
    """
    cette fonction prend en entrée une np.array de dimension 304x5 (les 5 courbes post processed prédites par le modèle)
    et renvoie une str: la classe de l'échantillon
    peut marcher à la fois pour les annotatinos réelles (ground truth) et les prédictions (si seuillées)
    """
    sample_class = "Unknown"
    # on détermine les deltas
    # par exemple pour une section de courbe:
    #  0  0  1  1  1  1  0  0
    # on va obtenir les nouvelles valeurs suivantes :
    #     0  1  0  0  0 -1  0
    # on a donc une nouvelle courbe (pour chaque dimension, i.e. G, A, M, k, l)
    # qui nous renseigne sur l'entrée/sortie d'un pic:
    # valeur==1 : on rentre dans un pic à cette position de la courbe
    # valeur==-1 : on sort d'un pic à cette position de la courbe
    # valeur==0 : on reste dans/hors d'un pic, pas de changement à cette position
    delta_curves = np.diff(postprocessed_predicted_curves, axis=0)
    # à partir de ces deltas, pour chaque dimension, on peut déterminer la position à laquelle on entre dans un pic
    mspikes_start_positions = [np.where(delta_curves[...,d]==1)[0] for d in range(delta_curves.shape[-1])]
    # on en déduit, pour chaque dimension, le nombre de pics retrouvés
    mspikes_numbers = [len(m) for m in mspikes_start_positions]
    # plusieurs cas de figures à ce stade:
    #    aucun pic sur la courbe : tous les 'mspikes_numbers' sont ==0
    #    1 pic sur la courbe :
    #        1 ou plusieurs 'mspikes_numbers' sont == 1 et toutes les valeurs 'mspikes_start_positions' sont
    #        égales à 0 ou la même valeur (i.e. les pics commencent au même endroit)
    #    plusieurs pics sur la courbe :
    #        plusieurs 'mspikes_numbers' sont == 1 et toutes les valeurs 'mspikes_start_positions' qui
    #        ne sont pas égales à zéro ont au moins 2 valeurs différentes (par ex: un pic commence à 200, et un autre commence à 240)
    # on va analyser ces différents cas de figure
    if all([m == 0 for m in mspikes_numbers]):
        # aucun pic: nombre de mspike==0 pour toutes les dimensions
        sample_class = "Normal"
    else: # au moins un pic détecté
        # on va faire la liste de toutes les "start positions" détectées, pour toutes les dimensions (G, A, M, k, l)
        start_positions_detected = [pos for m in mspikes_start_positions for pos in m]
        # on commence par voir s'il n'y aurait qu'une seule dimension (i.e. chaîne) affectée
        if len(start_positions_detected) == 1:
            # c'est le cas : juste un pic, donc probablement une chaîne légère libre ?
            # on vérifie qu'il n'y a pas de chaîne lourde, comme attendu
            if all(np.array(mspikes_numbers[:3])==0):
                # ok, c'est donc bien une chaîne légère libre
                light_concerned = np.where(np.array(mspikes_numbers[3:])==1)[0][0]
                sample_class = "Free "+("k","l")[light_concerned]
            else:
                # il n'y a qu'une seule courbe affectée, et c'est une chaîne lourde -> ce n'est pas normal !
                sample_class = "Complex (unsure of light clonal chain)"
        else:
            # sinon, on va déterminer si cette "position" est toujours la même sur les différentes chaînes
            all_mspikes_start_at_same_position = all([pos == start_positions_detected[0] for pos in start_positions_detected[1:]])
            if all_mspikes_start_at_same_position:
                # tous les pics commencent au même endroit, donc il n'y a qu'un seul pic monoclonal
                # il ne nous reste plus qu'à déterminer de quel type de chaînes il s'agit
                heavy_concerned = np.where(np.array(mspikes_numbers[:3])>0)[0]
                light_concerned = np.where(np.array(mspikes_numbers[3:])>0)[0]
                if (len(heavy_concerned) > 1) | (len(light_concerned) > 1):
                    # plusieurs chaînes lourdes ou légères pour le même pic ? probablement une erreur
                    sample_class = "Complex (multiple clonal chains detected, at same location?)"
                elif (len(heavy_concerned) == 0) & (len(light_concerned) == 1): # pas de chaine lourde mais une chaîne légère -> pic à chaîne légère libre
                    sample_class = "Free "+("k","l")[light_concerned[0]]
                elif (len(heavy_concerned) == 1) & (len(light_concerned) == 1): # 1 chaîne lourde, 1 chaîne légère -> cas classique
                    sample_class = "Ig "+("G","A","M")[heavy_concerned[0]]+("k","l")[light_concerned[0]]
                else: # normalement on ne devrait pas arriver là (sauf si heavy==0 et light==0)
                    # mais on prévoit le coup au cas où
                    sample_class = "Error 1"
            else:
                # il y a visiblement plusieurs pics
                # on va déterminer s'il s'agit toujours de la même Ig
                # ou s'il s'agit au moins une fois d'une Ig différente
                # pour cela, on va simplement comparer les courbes G, A, M, k, l qui ne sont pas remplies de zéros entre elles
                # si elles sont identiques : c'est toujours la même Ig
                # sinon : le pic n'est pas toujours composé des mêmes chaînes
                ref_mspikes_loc = None
                all_affected_chains_are_identical=True
                for d in range(len(mspikes_start_positions)): # pour chaque courbe (i.e., dimension, G, A, M, k, l)
                    if len(mspikes_start_positions[d])==0:
                        continue # pas de pic sur cette courbe -> on continue
                    else: # >= 1 pic a été trouvé
                        if ref_mspikes_loc is None:
                            # on avait pas encore trouvé de pic
                            ref_mspikes_loc = mspikes_start_positions[d]
                        else: # on avait déjà trouvé une chaîne avec 1/des pics
                            if len(ref_mspikes_loc) == len(mspikes_start_positions[d]):
                                if any(ref_mspikes_loc != mspikes_start_positions[d]):
                                    # les pics ne sont pas à la même position sur cette chaîne!
                                    all_affected_chains_are_identical = False
                                    break
                            else:
                                all_affected_chains_are_identical = False
                                break
                if all_affected_chains_are_identical:
                    # un seul "type" de pics : on peut la classer dans une catégorie "basique"
                    heavy_concerned = np.where(np.array(mspikes_numbers[:3])>0)[0]
                    light_concerned = np.where(np.array(mspikes_numbers[3:])>0)[0]
                    if (len(heavy_concerned) > 0) & (len(light_concerned) > 0): # pics à Ig entière
                        sample_class = "Ig "+("G","A","M")[heavy_concerned[0]]+("k","l")[light_concerned[0]]
                    elif (len(heavy_concerned) == 0) & (len(light_concerned) > 0): # pas de chaine lourde mais >=1 chaîne légère -> pics à chaîne légère libre
                        sample_class = "Free "+("k","l")[light_concerned[0]]
                    else: # normalement on ne devrait pas arriver là (heavy>0 et light==0 ?)
                        # mais on prévoit le coup au cas où
                        sample_class = "Error 2"
                else:
                    # plusieurs types de pics : on va la classer en "complexe"
                    sample_class = "Complex"
    return sample_class
    
sample_gt_classes = []
sample_pred_classes = []
# the threshold used for the predictions
threshold=.5

# pour chaque courbe on calcule
for i in tqdm(range(y_test.shape[0])):
    # la classe réelle
    sample_gt_classes.append(classifySample(y_test[i,...]))
    # la classe prédite
    sample_pred_classes.append(classifySample((y_test_[i,...]>threshold)*1))

sample_gt_classes = np.array(sample_gt_classes)
sample_pred_classes = np.array(sample_pred_classes)

# We can write a simple cross tab with pandas
pd.crosstab(sample_gt_classes, sample_pred_classes)

# Or see below for a more detailed and pretty presentation of the results into a .xlsx file

#### Export results as a pretty xlsx sheet ####

# the classes that may be found for both ground truth and predicted annotations
possible_classes = ('Ig Gk', 'Ig Gl', 'Ig Ak', 'Ig Al', 'Ig Mk', 'Ig Ml',
                    'Free k', 'Free l',
                    'Complex',
                    'Normal',)
# the classes that may only be found for predicted annotations (potential errors? the NN was unable to correctly predict those samples)
possible_classes_pred_only = ('Complex (multiple clonal chains detected, at same location?)',
                              'Complex (unsure of light clonal chain)',
                              'Error 1',
                              'Error 2',
                              'Unknown')

#####
#####
EXCEL_FILE_NAME = file[:-3]+"xlsx"
# EXCEL_FILE_NAME = "Accuracy table 3D.xlsx"
#####
#####

writer = pd.ExcelWriter(os.path.join(PATH_OUT,EXCEL_FILE_NAME), engine='xlsxwriter')
pd.DataFrame().to_excel(writer, sheet_name='Accuracy')
workbook  = writer.book
worksheet = writer.sheets['Accuracy']

# write row/column names
fmt_bold = workbook.add_format({'bold': True, 'font_color': 'black'})
worksheet.set_column('A:A', 15)
worksheet.set_column('B:'+chr(65+len(possible_classes)), 8)
worksheet.set_column(chr(65+len(possible_classes)+1)+':'+chr(65+len(possible_classes)+1), 2)
worksheet.set_column(chr(65+len(possible_classes)+2)+':'+chr(65+len(possible_classes)+2), 12)
worksheet.set_column(chr(65+len(possible_classes)+3)+':'+chr(65+len(possible_classes)+3), 2)
worksheet.set_column(chr(65+len(possible_classes)+4)+':'+chr(65+len(possible_classes)+4+len(possible_classes_pred_only)), 8)
worksheet.write('A1', "Predicted ->", fmt_bold)
worksheet.write(chr(65+len(possible_classes)+2)+'1', "Recall", fmt_bold)
worksheet.write('A'+str(len(possible_classes)+3), "Precision", fmt_bold)
worksheet.write(chr(65+len(possible_classes)-1)+str(len(possible_classes)+5), "Global accuracy:", fmt_bold)
worksheet.write(chr(65+len(possible_classes)+len(possible_classes_pred_only)+2)+str(len(possible_classes)+5), "Uninterpretable:", fmt_bold)
for col,name in enumerate(possible_classes):
    worksheet.write(chr(65+col+1)+'1', name, fmt_bold)
for row,name in enumerate(possible_classes):
    worksheet.write('A'+str(row+2), name, fmt_bold)
for col,name in enumerate(possible_classes_pred_only):
    worksheet.write(chr(65+col+len(possible_classes)+4+1)+'1', name, fmt_bold)

# write contingency table values
colors_diag = list(Color("#d2ffd2").range_to(Color("#228d20"),256))
colors_notdiag = list(Color("#ffffff").range_to(Color("#8d2020"),256))
for row,gt_class in enumerate(possible_classes):
    for col,pred_class in enumerate(possible_classes):
        # compute value at position in contingency color
        val = np.sum((sample_gt_classes == gt_class) & (sample_pred_classes == pred_class))
        # compute color
        if gt_class == pred_class:
            # tend to green if high in percentage
            if np.sum((sample_gt_classes == gt_class) | (sample_pred_classes == gt_class)) > 0 :
                pct = val / np.sum((sample_gt_classes == gt_class) | (sample_pred_classes == gt_class))
            else:
                pct = 0
            color_val = int(pct*255)
            if color_val==0 and pct>0:
                color_val=1
            new_cell_format = workbook.add_format({'bg_color': colors_diag[color_val].hex_l})
            worksheet.write(chr(65+col+1)+str(row+2), val, new_cell_format)
            # debug
            # worksheet.write(chr(65+col+1)+str(row+2+12), color_val)
        else:
            if np.sum((sample_gt_classes == gt_class)) > 0:
                pct = val / np.sum((sample_gt_classes == gt_class))
            else:
                pct = 0
            color_val = int(pct*255)
            if color_val==0 and pct>0:
                color_val=1
            new_cell_format = workbook.add_format({'bg_color': colors_notdiag[color_val].hex_l})
            worksheet.write(chr(65+col+1)+str(row+2), val, new_cell_format)
            # debug
            # worksheet.write(chr(65+col+1)+str(row+2+12), color_val)
            
# continue contingency table values, only for classes "predicted"
for row,gt_class in enumerate(possible_classes):
    for col,pred_class in enumerate(possible_classes_pred_only):
        # compute value at position in contingency color
        val = np.sum((sample_gt_classes == gt_class) & (sample_pred_classes == pred_class))
        # compute color
        if np.sum((sample_gt_classes == gt_class)) > 0:
            pct = val / np.sum((sample_gt_classes == gt_class))
        else:
            pct = 0
        color_val = int(pct*255)
        if color_val==0 and pct>0:
            color_val=1
        new_cell_format = workbook.add_format({'bg_color': colors_notdiag[color_val].hex_l})
        worksheet.write(chr(65+col+len(possible_classes)+4+1)+str(row+2), val, new_cell_format)
        # debug
        # worksheet.write(chr(65+col+1)+str(row+2+12), color_val)

# display accuracies by ground truth class
fmt_percentage = workbook.add_format({'num_format': '0.0%'})  
for row,name in enumerate(possible_classes):
    print_col = chr(65+len(possible_classes)+2)
    start_col = 'B'
    true_pos_col = chr(65+row+1)
    end_col = chr(65+len(possible_classes))
    print_row = str(row+2)
    start_row = str(row+2)
    true_pos_row = str(row+2)
    end_row = str(row+2)
    print_cell = print_col+print_row
    print_formula = '=IF(SUM({}{}:{}{})>0,({}{}/SUM({}{}:{}{})),"")'.format(start_col,start_row,end_col,end_row,true_pos_col,true_pos_row,start_col,start_row,end_col,end_row)
    worksheet.write_formula(print_cell, print_formula, cell_format = fmt_percentage)
# same for accuracy by predicted class
for col,name in enumerate(possible_classes):
    print_col = chr(65+col+1)
    start_col = chr(65+col+1)
    true_pos_col = chr(65+col+1)
    end_col = chr(65+col+1)
    print_row = str(len(possible_classes)+3)
    start_row = str(2)
    true_pos_row = str(col+2)
    end_row = str(len(possible_classes)+1)
    print_cell = print_col+print_row
    print_formula = '=IF(SUM({}{}:{}{})>0,({}{}/SUM({}{}:{}{})),"")'.format(start_col,start_row,end_col,end_row,true_pos_col,true_pos_row,start_col,start_row,end_col,end_row)
    worksheet.write_formula(print_cell, print_formula, cell_format = fmt_percentage)

# format colors for accuracy values
worksheet.conditional_format(chr(65+len(possible_classes)+2)+'2:'+chr(65+len(possible_classes)+2)+str(len(possible_classes)+1), {'type': '3_color_scale'})
worksheet.conditional_format('B'+str(len(possible_classes)+3)+':'+chr(65+len(possible_classes))+str(len(possible_classes)+3), {'type': '3_color_scale'})

# global accuracy
accuracy_formula = "=("
first_iter=True
for v,gt_class in enumerate(possible_classes):
    if first_iter:
        first_iter=False
    else:
        accuracy_formula += '+'
    accuracy_formula += chr(65+v+1)+str(v+2)
accuracy_formula += ')/SUM(B2:'+chr(65+v+1)+str(v+2)+')'
accuracy_formula

fmt_percentage_bold = workbook.add_format({'num_format': '0.0%', 'bold': True, 'font_color': 'black'})
worksheet.write(chr(65+len(possible_classes)+2)+str(len(possible_classes)+5), accuracy_formula, fmt_percentage_bold)

primary_cont_table_locs = 'B2:'+chr(65+len(possible_classes))+str(len(possible_classes)+1)
second_cont_table_locs = chr(65+len(possible_classes)+5)+'2:'+chr(65+len(possible_classes)+len(possible_classes_pred_only)+4)+str(len(possible_classes)+1)
uninterpret_formula = '=SUM({})/(SUM({})+SUM({}))'.format(second_cont_table_locs,second_cont_table_locs,primary_cont_table_locs)

# finally compute percentage of uninterpretable samples:
worksheet.write(chr(65+len(possible_classes)+len(possible_classes_pred_only)+4)+str(len(possible_classes)+5), uninterpret_formula, fmt_percentage)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

    

    
    
    
    

    
    
    
    

    
    
    
    
    
    
    