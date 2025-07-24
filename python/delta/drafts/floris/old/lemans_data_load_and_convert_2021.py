# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:41:53 2019

@author: Floris Chabrun
"""

print('Starting segmentation training script...')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--model_name", type=str, default='segmentation_spikes_best_full_model_2020-batchsize-32.h5')
FLAGS = parser.parse_args()

model_name = FLAGS.model_name

debug_mode = FLAGS.debug
separate_peak_fraction = False

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# import matplotlib.lines as ml

if debug_mode == 1:
    path_in = 'C:/Users/admin/Documents/Capillarys/data/data/full2020/output'
    path_out = 'C:/Users/admin/Documents/Capillarys/temp2020'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectr/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectr/out'

# load raw data
raw = pd.read_csv(r"C:\Users\admin\Documents\Capillarys\data\2021\ifs\lemans_if_matv1.csv")

# on définit la taille d'une courbe
spe_width=304

curve_columns = [c for c in raw.columns if c[:4]=='ELPx']
Kcurve_columns = [c for c in raw.columns if c[:2]=='Kx']
Lcurve_columns = [c for c in raw.columns if c[:2]=='Lx']
Gcurve_columns = [c for c in raw.columns if c[:4]=='IgGx']
Acurve_columns = [c for c in raw.columns if c[:4]=='IgAx']
Mcurve_columns = [c for c in raw.columns if c[:4]=='IgMx']
annot_columns = [c for c in raw.columns if c[:2]=='ig']
len(curve_columns) # should be 304 for all
len(Kcurve_columns)
len(Lcurve_columns)
len(Gcurve_columns)
len(Acurve_columns)
len(Mcurve_columns)
len(annot_columns)

x=raw.loc[:,curve_columns].to_numpy()

# normalize
x = x/(np.max(x, axis = 1)[:,None])

# other curves
xK=raw.loc[:,Kcurve_columns].to_numpy()
xL=raw.loc[:,Lcurve_columns].to_numpy()
xG=raw.loc[:,Gcurve_columns].to_numpy()
xA=raw.loc[:,Acurve_columns].to_numpy()
xM=raw.loc[:,Mcurve_columns].to_numpy()

xK = xK/(np.max(xK, axis = 1)[:,None])
xL = xL/(np.max(xL, axis = 1)[:,None])
xG = xG/(np.max(xG, axis = 1)[:,None])
xA = xA/(np.max(xA, axis = 1)[:,None])
xM = xM/(np.max(xM, axis = 1)[:,None])

annot = raw.loc[:,annot_columns].to_numpy()

# %%

# load model
model=tf.keras.models.load_model(os.path.join(path_out,model_name), compile=False)

print(model.summary())
    
# Make predictions and see those predictions
import time

size=x.shape[0]
start=time.time()
y_=model.predict(x[:size,...].reshape(size, spe_width, 1, 1))
end=time.time()
print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms')
print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s')

y_annotations_unchecked = y_[:,:,0,0]
np.save(r"C:\Users\admin\Documents\Capillarys\data\2021\ifs\lemans_if_annot_unchecked.npy", y_annotations_unchecked)

# %%

y_annotations_unchecked = np.load(r"C:\Users\admin\Documents\Capillarys\data\2021\ifs\lemans_if_annot_unchecked.npy")

detection_threshold = 0.5

# compute class maps
class_maps=(y_annotations_unchecked>detection_threshold)*1
# on force les points 0-150 à être négatifs
class_maps[:,:150] = 0


# np.sum(np.sum(y_[:,:100,0,0]>.5, axis=1)>0) # 1638 curves with spike in 100 first points
# np.sum(np.sum(y_[:,100:200,0,0]>.5, axis=1)>0) # 163 curves with spike in 100 first points
# np.sum(np.sum(y_[:,200:,0,0]>.5, axis=1)>0) # 1736 curves with spike in 100 first points
# le modèle identifie quasi-toujours un pic au début de l'albumine
# on va donc l'empêcher de placer des pics dans les 150 premiers points (i.e. alb, a1, début d'a2)

# Plot a curve from the validation set:
def plotSpikesMap(ix):
    annot_columns_display = ['IgGk','IgGl','IgAk','IgAl','IgMk','IgMl']
    text_annot = ""
    if np.sum(annot[ix])==0:
        text_annot = "Ground truth: NO Ig detected on IT"
    else:
        text_annot = "Ground truth: " + "/".join([annot_columns_display[i] for i in np.where(annot[ix]==1)[0]])
    plt.figure(figsize=(14,10))
    plt.subplot(2,1,1)
    plt.text(spe_width-1,1,"Curve {} // ".format(ix)+text_annot, horizontalalignment="right")
    # on plotte la map de prédictions
    plt.plot(np.arange(0,spe_width), (y_annotations_unchecked[ix,:]), '-', color='blue')
    # on récupère la class map (binarisée)
    class_map=class_maps[ix,:]
    curve_values = x[ix,:]
    plt.plot(np.arange(0,spe_width), curve_values, '-', color = 'black')
    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

    # on plot aussi les autres courbes
    plt.subplot(2,1,2)
    plt.text(250,1,"Curve {} // ".format(ix)+text_annot, horizontalalignment="right")
    plt.plot(np.arange(0,304)+1, xK[ix,:], '-', color = 'red', label = "k")
    plt.plot(np.arange(0,304)+1, xL[ix,:], '-', color = 'blue', label = "l")
    plt.plot(np.arange(0,304)+1, xG[ix,:], '-', color = 'purple', label = "G")
    plt.plot(np.arange(0,304)+1, xA[ix,:], '-', color = 'pink', label = "A")
    plt.plot(np.arange(0,304)+1, xM[ix,:], '-', color = 'green', label = "M")
    plt.legend()

# debug     -> affichage d'une courbe
# plt.ion()
# plotSpikesMap(1)

# %%

# on va sauvegarder une image de chaque IF
# et on va manuellement les trier
# pour voir celles pour lesquels il y a besoin d'une modification sur l'annotation
# save all
output_path = r"C:\Users\admin\Documents\Capillarys\data\test2020\ifs_attendingsort"

plt.ioff()
for i in tqdm(range(x.shape[0])):
    plotSpikesMap(i)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,"{}.jpg".format(i)))

# Une fois ce tri effectué, on va ci dessous lister les modifications à réaliser pour chaque courbe
plt.ion()

# %%

run_quickly = True
if run_quickly:
    def plotSpikesMap(ix):
        return None

curve_id = 10
plotSpikesMap(curve_id) # uniquement le 1er pic = un vrai pic monoclonal IgGl # le 2e est à supprimer
class_maps[curve_id,232:250] = 0
plotSpikesMap(curve_id) # ok

curve_id = 12
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,224:231] = 1
plotSpikesMap(curve_id) # ok

curve_id = 59
plotSpikesMap(curve_id) # pic en fin de gamma et pas beta
class_maps[curve_id,:] = 0
class_maps[curve_id,241:247] = 1
plotSpikesMap(curve_id)

curve_id = 141
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,249:259] = 1
plotSpikesMap(curve_id)

curve_id = 163
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,202:209] = 1
plotSpikesMap(curve_id)

curve_id = 181
plotSpikesMap(curve_id)
# class_maps[curve_id,:] = 0
class_maps[curve_id,184:192] = 1
plotSpikesMap(curve_id)

curve_id = 261
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,180:187] = 1
plotSpikesMap(curve_id)

curve_id = 265
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,217:225] = 1
plotSpikesMap(curve_id)

curve_id = 276
plotSpikesMap(curve_id)
# class_maps[curve_id,:] = 0
class_maps[curve_id,229:240] = 0
plotSpikesMap(curve_id)

curve_id = 285
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,205:220] = 1
plotSpikesMap(curve_id)

curve_id = 333
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,211:218] = 1
plotSpikesMap(curve_id)

curve_id = 354
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,219:232] = 1
plotSpikesMap(curve_id)

curve_id = 359
plotSpikesMap(curve_id)
class_maps[curve_id,227:] = 0
plotSpikesMap(curve_id)

curve_id = 369
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,197:219] = 1
plotSpikesMap(curve_id)

curve_id = 470
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,221:227] = 1
class_maps[curve_id,228:237] = 1
class_maps[curve_id,239:246] = 1
plotSpikesMap(curve_id)

curve_id = 643
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,208:217] = 1
class_maps[curve_id,218:225] = 1
plotSpikesMap(curve_id)

curve_id = 644
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,226:235] = 1
class_maps[curve_id,236:245] = 1
class_maps[curve_id,247:252] = 1
plotSpikesMap(curve_id)

curve_id = 795
plotSpikesMap(curve_id)
class_maps[curve_id,226:236] = 0
plotSpikesMap(curve_id)

curve_id = 865
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,199:207] = 1
plotSpikesMap(curve_id)

curve_id = 872
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,239:246] = 1
plotSpikesMap(curve_id)

curve_id = 896
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,163:181] = 1
class_maps[curve_id,183:190] = 1
plotSpikesMap(curve_id)

curve_id = 900
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,217:224] = 1
plotSpikesMap(curve_id)

curve_id = 928
plotSpikesMap(curve_id)
class_maps[curve_id,217:222] = 1
class_maps[curve_id,233:239] = 1
plotSpikesMap(curve_id)

curve_id = 931
plotSpikesMap(curve_id)
class_maps[curve_id,:] = 0
class_maps[curve_id,227:235] = 1
plotSpikesMap(curve_id)



curve_id = 1389
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,188:200] = 1
class_maps[curve_id,209:219] = 1
plotSpikesMap(curve_id)

curve_id = 1417
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,200:211] = 1
class_maps[curve_id,219:227] = 1
plotSpikesMap(curve_id)

curve_id = 1453
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,225:] = 0
plotSpikesMap(curve_id)

curve_id = 1501
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,210:217] = 1
plotSpikesMap(curve_id)

curve_id = 1506
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,215:222] = 1
plotSpikesMap(curve_id)

curve_id = 1560
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,210:222] = 1
class_maps[curve_id,226:233] = 1
class_maps[curve_id,234:241] = 1
plotSpikesMap(curve_id)

curve_id = 1563
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,235:241] = 1
plotSpikesMap(curve_id)

curve_id = 1580
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,214:220] = 1
plotSpikesMap(curve_id)

curve_id = 1688
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,207:213] = 1
plotSpikesMap(curve_id)

curve_id = 1781
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,164:182] = 1
class_maps[curve_id,186:192] = 1
plotSpikesMap(curve_id)



curve_id = 1164
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,224:232] = 1
plotSpikesMap(curve_id)

curve_id = 1222
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,213:221] = 1
plotSpikesMap(curve_id)

curve_id = 1223
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,233:244] = 1
plotSpikesMap(curve_id)

curve_id = 1226
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,242:251] = 1
plotSpikesMap(curve_id)

curve_id = 1236
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,203:217] = 1
plotSpikesMap(curve_id)


curve_id = 1124
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,213:222] = 1
plotSpikesMap(curve_id)

curve_id = 1147
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,238:246] = 1
plotSpikesMap(curve_id)

curve_id = 976
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,224:235] = 1
plotSpikesMap(curve_id)

curve_id = 1035
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,204:218] = 1
class_maps[curve_id,228:236] = 1
plotSpikesMap(curve_id)

curve_id = 542
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,218:228] = 1
plotSpikesMap(curve_id)

curve_id = 605
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,225:234] = 1
class_maps[curve_id,242:250] = 1
plotSpikesMap(curve_id)

curve_id = 852
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:218] = 1
plotSpikesMap(curve_id)



# %%

# Attention !
# Il n'y a pas de chaînes légères libres seules ! Uniquement des anticorps lourdes + légères

# On convertit la class_map en une dimension en class_map en dimensions multiples
extended_class_maps = np.zeros(class_maps.shape + (5,)) # 5 pour g,a,m,k,l
untreated_samples = []
for i in range(class_maps.shape[0]):
    if np.sum(annot[i])==1:
        # simple : tous les pics de la courbe = l'ig détectée
        ig_anomaly = annot_columns[np.where(annot[i])[0][0]]
        if ig_anomaly[:3]=="igg":
            extended_class_maps[i,:,0] = class_maps[i]
        elif ig_anomaly[:3]=="iga":
            extended_class_maps[i,:,1] = class_maps[i]
        elif ig_anomaly[:3]=="igm":
            extended_class_maps[i,:,2] = class_maps[i]
        if ig_anomaly[-1:]=="k":
            extended_class_maps[i,:,3] = class_maps[i]
        elif ig_anomaly[-1:]=="l":
            extended_class_maps[i,:,4] = class_maps[i]
    elif np.sum(annot[i])>1:
        # à faire manuellement : plusieurs types d'Ig
        untreated_samples.append(i)
    # pas de else : si pas d'annotation, pas de pic à marquer
    
untreated_samples # quels éléments n'ont pas été traités (automatiquement ?) # 73

# %%

# vérifions sur certaines courbes que tout s'est bien passé
# on va faire un visualiseur
def plotExtendedIFMaps(ix):
    annot_columns_display = ['IgGk','IgGl','IgAk','IgAl','IgMk','IgMl']
    text_annot = ""
    if np.sum(annot[ix])==0:
        text_annot = "Ground truth: NO Ig detected on IT"
    else:
        text_annot = "Ground truth: " + "/".join([annot_columns_display[i] for i in np.where(annot[ix]==1)[0]])
    class_map=class_maps[ix,:]
    curve_values = x[ix,:]
    x_max = np.argmax(curve_values)
    plt.figure(figsize=(14,12))
    plt.subplot(6,2,1)
    plt.text(spe_width-1,1,"Curve {} // ".format(ix)+text_annot, horizontalalignment="right")
    plt.plot(np.arange(0,spe_width), (y_annotations_unchecked[ix,:]), '-', color='blue')
    plt.plot(np.arange(0,spe_width), curve_values, '-', color = 'black')
    plt.text(0,1, 'Reference', verticalalignment='top')
    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')
    plt.text(x_max+5,.5,x_max, verticalalignment='top', color='red')
        
    plt.subplot(6,2,2)
    plt.text(spe_width-1,1,"Curve {} // ".format(ix)+text_annot, horizontalalignment="right")
    plt.plot(np.arange(0,spe_width), (y_annotations_unchecked[ix,:]), '-', color='blue')
    plt.plot(np.arange(0,spe_width), curve_values, '-', color = 'black')
    plt.text(0,1, 'Reference', verticalalignment='top')
    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')
    plt.text(x_max+5,.5,x_max, verticalalignment='top', color='red')

    annot_ig_display = ['G','A','M','K','L']
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
        plt.text(0,1,annot_ig_display[ig_plane], verticalalignment='top')
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

# plotExtendedIFMaps(0)
# plotExtendedIFMaps(900) # cela a l'air de bien marcher

# on va maintenant pouvoir corriger les courbes avec 2 ig différentes

# %%

run_quickly = True
if run_quickly:
    def plotSpikesMap(ix):
        return None
    def plotExtendedIFMaps(ix):
        return None

treated_samples = []

curve_id = 51
plotSpikesMap(curve_id) # 2 pics : Mk puis Gk
np.where(class_maps[curve_id,:]==1) # 218 à 234
extended_class_maps[curve_id,218:228,2] = 1
extended_class_maps[curve_id,218:228,-2] = 1
extended_class_maps[curve_id,228:235,0] = 1
extended_class_maps[curve_id,228:235,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 342
plotSpikesMap(curve_id) # 2 pics : Al puis Gl
np.where(class_maps[curve_id,:]==1)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,210:224,1] = 1
extended_class_maps[curve_id,210:224,-1] = 1
extended_class_maps[curve_id,236:248,0] = 1
extended_class_maps[curve_id,236:248,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 390
plotSpikesMap(curve_id) # 2 pics : Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,218:228] = 1 # Mk
class_maps[curve_id,240:251] = 1 # Gk
plotSpikesMap(curve_id) # 2 pics : Mk puis Gk
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,218:228,2] = 1
extended_class_maps[curve_id,218:228,-2] = 1
extended_class_maps[curve_id,240:251,0] = 1
extended_class_maps[curve_id,240:251,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 453
plotSpikesMap(curve_id) # 2 pics : Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,219:228] = 1
class_maps[curve_id,229:239] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,219:228,0] = 1
extended_class_maps[curve_id,219:228,-1] = 1
extended_class_maps[curve_id,229:239,0] = 1
extended_class_maps[curve_id,229:239,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 512
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,206:213] = 1
class_maps[curve_id,233:246] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,206:213,0] = 1
extended_class_maps[curve_id,206:213,-1] = 1
extended_class_maps[curve_id,233:246,0] = 1
extended_class_maps[curve_id,233:246,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 590
plotSpikesMap(curve_id) # 2 pics: Mk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,217:226] = 1
class_maps[curve_id,229:236] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,217:226,2] = 1
extended_class_maps[curve_id,217:226,-2] = 1
extended_class_maps[curve_id,229:236,0] = 1
extended_class_maps[curve_id,229:236,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 610
plotSpikesMap(curve_id) # 2 pics: Ak puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:222] = 1
class_maps[curve_id,232:242] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,209:222,1] = 1
extended_class_maps[curve_id,209:222,-2] = 1
extended_class_maps[curve_id,232:242,0] = 1
extended_class_maps[curve_id,232:242,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 620
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,219:225] = 1
class_maps[curve_id,231:240] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,219:225,0] = 1
extended_class_maps[curve_id,219:225,-2] = 1
extended_class_maps[curve_id,231:240,0] = 1
extended_class_maps[curve_id,231:240,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 683
plotSpikesMap(curve_id) # 2 pics: Gk puis Al
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,221:230] = 1
class_maps[curve_id,235:247] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,221:230,0] = 1
extended_class_maps[curve_id,221:230,-2] = 1
extended_class_maps[curve_id,235:247,1] = 1
extended_class_maps[curve_id,235:247,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 688
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,225:236] = 1
class_maps[curve_id,237:246] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,225:236,2] = 1
extended_class_maps[curve_id,225:236,-2] = 1
extended_class_maps[curve_id,237:246,0] = 1
extended_class_maps[curve_id,237:246,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 746
plotSpikesMap(curve_id) # 2 pics: Ak puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,210:227] = 1
class_maps[curve_id,233:243] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,210:227,1] = 1
extended_class_maps[curve_id,210:227,-2] = 1
extended_class_maps[curve_id,233:243,0] = 1
extended_class_maps[curve_id,233:243,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 764
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,214:222] = 1
class_maps[curve_id,230:244] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,214:222,2] = 1
extended_class_maps[curve_id,214:222,-2] = 1
extended_class_maps[curve_id,230:244,0] = 1
extended_class_maps[curve_id,230:244,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 772
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,216:225] = 1
class_maps[curve_id,226:235] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,216:225,2] = 1
extended_class_maps[curve_id,216:225,-2] = 1
extended_class_maps[curve_id,226:235,0] = 1
extended_class_maps[curve_id,226:235,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 790
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,204:220] = 1
class_maps[curve_id,223:232] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,204:220,0] = 1
extended_class_maps[curve_id,204:220,-2] = 1
extended_class_maps[curve_id,223:232,0] = 1
extended_class_maps[curve_id,223:232,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 859
plotSpikesMap(curve_id) # 2 pics: Mk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,214:226] = 1
class_maps[curve_id,228:236] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,214:226,2] = 1
extended_class_maps[curve_id,214:226,-2] = 1
extended_class_maps[curve_id,228:236,0] = 1
extended_class_maps[curve_id,228:236,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 863
plotSpikesMap(curve_id) # 2 pics: Gk puis Ml
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,204:215] = 1
class_maps[curve_id,217:225] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,204:215,0] = 1
extended_class_maps[curve_id,204:215,-2] = 1
extended_class_maps[curve_id,217:225,2] = 1
extended_class_maps[curve_id,217:225,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 866
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,201:210] = 1
class_maps[curve_id,222:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,201:210,2] = 1
extended_class_maps[curve_id,201:210,-2] = 1
extended_class_maps[curve_id,222:231,0] = 1
extended_class_maps[curve_id,222:231,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 892
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,225:232] = 1
class_maps[curve_id,233:242] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,225:232,0] = 1
extended_class_maps[curve_id,225:232,-1] = 1
extended_class_maps[curve_id,233:242,0] = 1
extended_class_maps[curve_id,233:242,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1314
plotSpikesMap(curve_id) # 2 pics: Al puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,191:198] = 1
class_maps[curve_id,227:236] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,191:198,1] = 1
extended_class_maps[curve_id,191:198,-1] = 1
extended_class_maps[curve_id,227:236,2] = 1
extended_class_maps[curve_id,227:236,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1357
plotSpikesMap(curve_id) # 2 pics: Ak puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,184:197] = 1
class_maps[curve_id,214:225] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,184:197,1] = 1
extended_class_maps[curve_id,184:197,-2] = 1
extended_class_maps[curve_id,214:225,0] = 1
extended_class_maps[curve_id,214:225,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1410
plotSpikesMap(curve_id) # 2 pics: Al puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,192:205] = 1
class_maps[curve_id,218:226] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,192:205,1] = 1
extended_class_maps[curve_id,192:205,-1] = 1
extended_class_maps[curve_id,218:226,0] = 1
extended_class_maps[curve_id,218:226,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1466
plotSpikesMap(curve_id) # 2 pics: Gk puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,186:195] = 1
class_maps[curve_id,221:230] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,186:195,0] = 1
extended_class_maps[curve_id,186:195,-2] = 1
extended_class_maps[curve_id,221:230,2] = 1
extended_class_maps[curve_id,221:230,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1550
plotSpikesMap(curve_id) # 2 pics: Ml puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,225:237] = 1
class_maps[curve_id,238:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,225:237,2] = 1
extended_class_maps[curve_id,225:237,-1] = 1
extended_class_maps[curve_id,238:245,0] = 1
extended_class_maps[curve_id,238:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1611
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:216] = 1
class_maps[curve_id,226:234] = 1
class_maps[curve_id,237:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,209:216,2] = 1
extended_class_maps[curve_id,209:216,-2] = 1
extended_class_maps[curve_id,226:234,0] = 1
extended_class_maps[curve_id,226:234,-2] = 1
extended_class_maps[curve_id,237:245,0] = 1
extended_class_maps[curve_id,237:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1647
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,215:223] = 1
class_maps[curve_id,238:244] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,215:223,0] = 1
extended_class_maps[curve_id,215:223,-1] = 1
extended_class_maps[curve_id,238:244,0] = 1
extended_class_maps[curve_id,238:244,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1660
plotSpikesMap(curve_id) # 2 pics: Al puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,189:204] = 1
class_maps[curve_id,227:242] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,189:204,1] = 1
extended_class_maps[curve_id,189:204,-1] = 1
extended_class_maps[curve_id,227:242,0] = 1
extended_class_maps[curve_id,227:242,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1665
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,206:216] = 1
class_maps[curve_id,236:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,206:216,2] = 1
extended_class_maps[curve_id,206:216,-1] = 1
extended_class_maps[curve_id,236:245,2] = 1
extended_class_maps[curve_id,236:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1668
plotSpikesMap(curve_id) # 3 pics: Mk, Gk puis re-Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,208:217] = 1
class_maps[curve_id,218:225] = 1
class_maps[curve_id,229:239] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,208:217,2] = 1
extended_class_maps[curve_id,208:217,-2] = 1
extended_class_maps[curve_id,218:225,0] = 1
extended_class_maps[curve_id,218:225,-2] = 1
extended_class_maps[curve_id,229:239,2] = 1
extended_class_maps[curve_id,229:239,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1670
plotSpikesMap(curve_id) # 2 pics: Ml puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,206:215] = 1
class_maps[curve_id,224:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,206:215,2] = 1
extended_class_maps[curve_id,206:215,-1] = 1
extended_class_maps[curve_id,224:231,0] = 1
extended_class_maps[curve_id,224:231,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1685
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,208:218] = 1
class_maps[curve_id,221:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,208:218,0] = 1
extended_class_maps[curve_id,208:218,-2] = 1
extended_class_maps[curve_id,221:231,0] = 1
extended_class_maps[curve_id,221:231,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1691
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,216:226] = 1
class_maps[curve_id,228:236] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,216:226,2] = 1
extended_class_maps[curve_id,216:226,-2] = 1
extended_class_maps[curve_id,228:236,0] = 1
extended_class_maps[curve_id,228:236,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1705
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,218:230] = 1
class_maps[curve_id,234:243] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,218:230,2] = 1
extended_class_maps[curve_id,218:230,-1] = 1
extended_class_maps[curve_id,234:243,2] = 1
extended_class_maps[curve_id,234:243,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1716
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,220:230] = 1
class_maps[curve_id,234:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,220:230,2] = 1
extended_class_maps[curve_id,220:230,-2] = 1
extended_class_maps[curve_id,234:245,0] = 1
extended_class_maps[curve_id,234:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1736
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl, retirer pic au milieu
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,203:216] = 1
class_maps[curve_id,231:239] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,203:216,0] = 1
extended_class_maps[curve_id,203:216,-2] = 1
extended_class_maps[curve_id,231:239,0] = 1
extended_class_maps[curve_id,231:239,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1754
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,186:197] = 1
class_maps[curve_id,214:227] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,186:197,2] = 1
extended_class_maps[curve_id,186:197,-2] = 1
extended_class_maps[curve_id,214:227,0] = 1
extended_class_maps[curve_id,214:227,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1762
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,208:226] = 1
class_maps[curve_id,239:251] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,208:226,2] = 1
extended_class_maps[curve_id,208:226,-2] = 1
extended_class_maps[curve_id,239:251,0] = 1
extended_class_maps[curve_id,239:251,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1786
plotSpikesMap(curve_id) # 2 pics: Gl puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:221] = 1
class_maps[curve_id,228:240] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,209:221,0] = 1
extended_class_maps[curve_id,209:221,-1] = 1
extended_class_maps[curve_id,228:240,2] = 1
extended_class_maps[curve_id,228:240,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1797
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,197:207] = 1
class_maps[curve_id,213:220] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,197:207,0] = 1
extended_class_maps[curve_id,197:207,-1] = 1
extended_class_maps[curve_id,213:220,0] = 1
extended_class_maps[curve_id,213:220,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 6
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,221:231] = 1
class_maps[curve_id,238:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,221:231,0] = 1
extended_class_maps[curve_id,221:231,-1] = 1
extended_class_maps[curve_id,238:245,0] = 1
extended_class_maps[curve_id,238:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 16
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,198:213] = 1
class_maps[curve_id,214:222] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,198:213,0] = 1
extended_class_maps[curve_id,198:213,-1] = 1
extended_class_maps[curve_id,214:222,0] = 1
extended_class_maps[curve_id,214:222,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 22
plotSpikesMap(curve_id) # 2 pics: Al puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,205:217] = 1
class_maps[curve_id,248:257] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,205:217,1] = 1
extended_class_maps[curve_id,205:217,-1] = 1
extended_class_maps[curve_id,248:257,0] = 1
extended_class_maps[curve_id,248:257,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 33
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,205:218] = 1
class_maps[curve_id,235:244] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,205:218,0] = 1
extended_class_maps[curve_id,205:218,-2] = 1
extended_class_maps[curve_id,235:244,0] = 1
extended_class_maps[curve_id,235:244,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 100
plotSpikesMap(curve_id) # 2 pics: Ak puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,191:203] = 1
class_maps[curve_id,223:234] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,191:203,1] = 1
extended_class_maps[curve_id,191:203,-2] = 1
extended_class_maps[curve_id,223:234,0] = 1
extended_class_maps[curve_id,223:234,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 112
plotSpikesMap(curve_id) # 2 pics: Al puis Ml
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,223:228] = 1
class_maps[curve_id,229:241] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,223:228,1] = 1
extended_class_maps[curve_id,223:228,-1] = 1
extended_class_maps[curve_id,229:241,2] = 1
extended_class_maps[curve_id,229:241,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 315
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,192:203] = 1
class_maps[curve_id,213:226] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,192:203,2] = 1
extended_class_maps[curve_id,192:203,-1] = 1
extended_class_maps[curve_id,213:226,2] = 1
extended_class_maps[curve_id,213:226,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 462
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,215:223] = 1
class_maps[curve_id,229:241] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,215:223,2] = 1
extended_class_maps[curve_id,215:223,-1] = 1
extended_class_maps[curve_id,229:241,2] = 1
extended_class_maps[curve_id,229:241,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1173
plotSpikesMap(curve_id) # 2 pics: Ak puis Al
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,187:195] = 1
class_maps[curve_id,226:236] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,187:195,1] = 1
extended_class_maps[curve_id,187:195,-2] = 1
extended_class_maps[curve_id,226:236,1] = 1
extended_class_maps[curve_id,226:236,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1201
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,203:213] = 1
class_maps[curve_id,218:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,203:213,2] = 1
extended_class_maps[curve_id,203:213,-2] = 1
extended_class_maps[curve_id,218:231,0] = 1
extended_class_maps[curve_id,218:231,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1211
plotSpikesMap(curve_id) # 2 pics: Ml puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,223:237] = 1
class_maps[curve_id,239:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,223:237,2] = 1
extended_class_maps[curve_id,223:237,-1] = 1
extended_class_maps[curve_id,239:245,0] = 1
extended_class_maps[curve_id,239:245,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1220
plotSpikesMap(curve_id) # 2 pics: Al puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:219] = 1
class_maps[curve_id,243:255] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,209:219,1] = 1
extended_class_maps[curve_id,209:219,-1] = 1
extended_class_maps[curve_id,243:255,0] = 1
extended_class_maps[curve_id,243:255,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1241
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,229:239] = 1
class_maps[curve_id,240:248] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,229:239,2] = 1
extended_class_maps[curve_id,229:239,-2] = 1
extended_class_maps[curve_id,240:248,0] = 1
extended_class_maps[curve_id,240:248,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1260
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,220:232] = 1
class_maps[curve_id,232:243] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,220:232,2] = 1
extended_class_maps[curve_id,220:232,-1] = 1
extended_class_maps[curve_id,232:243,2] = 1
extended_class_maps[curve_id,232:243,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 981
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,214:221] = 1
class_maps[curve_id,223:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,214:221,2] = 1
extended_class_maps[curve_id,214:221,-2] = 1
extended_class_maps[curve_id,223:231,0] = 1
extended_class_maps[curve_id,223:231,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1007
plotSpikesMap(curve_id) # 2 pics: Al puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,200:209] = 1
class_maps[curve_id,219:229] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,200:209,1] = 1
extended_class_maps[curve_id,200:209,-1] = 1
extended_class_maps[curve_id,219:229,2] = 1
extended_class_maps[curve_id,219:229,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1088
plotSpikesMap(curve_id) # 3 pics: Al, Mk puis re-Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,179:191] = 1
class_maps[curve_id,214:224] = 1
class_maps[curve_id,236:243] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,179:191,1] = 1
extended_class_maps[curve_id,179:191,-1] = 1
extended_class_maps[curve_id,214:224,2] = 1
extended_class_maps[curve_id,214:224,-2] = 1
extended_class_maps[curve_id,236:243,2] = 1
extended_class_maps[curve_id,236:243,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1093
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,220:229] = 1
class_maps[curve_id,233:245] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,220:229,0] = 1
extended_class_maps[curve_id,220:229,-2] = 1
extended_class_maps[curve_id,233:245,0] = 1
extended_class_maps[curve_id,233:245,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1106
plotSpikesMap(curve_id) # 3 pics: Al, re-Al puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,194:208] = 1
class_maps[curve_id,216:224] = 1
class_maps[curve_id,244:255] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,194:208,1] = 1
extended_class_maps[curve_id,194:208,-1] = 1
extended_class_maps[curve_id,216:224,1] = 1
extended_class_maps[curve_id,216:224,-1] = 1
extended_class_maps[curve_id,244:255,0] = 1
extended_class_maps[curve_id,244:255,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1061
plotSpikesMap(curve_id) # 2 pics: Al puis cl lambda
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,193:212] = 1
class_maps[curve_id,220:231] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,193:212,1] = 1
extended_class_maps[curve_id,193:212,-1] = 1
# extended_class_maps[curve_id,220:231,1] = 1
extended_class_maps[curve_id,220:231,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 531
plotSpikesMap(curve_id) # 3 pics: Gl, Gk, Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,219:226] = 1
class_maps[curve_id,230:238] = 1
class_maps[curve_id,239:247] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,219:226,0] = 1
extended_class_maps[curve_id,219:226,-1] = 1
extended_class_maps[curve_id,230:238,0] = 1
extended_class_maps[curve_id,230:238,-2] = 1
extended_class_maps[curve_id,239:247,0] = 1
extended_class_maps[curve_id,239:247,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 543
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,223:232] = 1
class_maps[curve_id,234:243] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,223:232,0] = 1
extended_class_maps[curve_id,223:232,-1] = 1
extended_class_maps[curve_id,234:243,0] = 1
extended_class_maps[curve_id,234:243,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 564
plotSpikesMap(curve_id) # 2 pics: Gk puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,198:210] = 1
class_maps[curve_id,230:240] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,198:210,0] = 1
extended_class_maps[curve_id,198:210,-2] = 1
extended_class_maps[curve_id,230:240,2] = 1
extended_class_maps[curve_id,230:240,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 627
plotSpikesMap(curve_id) # 2 pics: Mk puis Ml
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,213:223] = 1
class_maps[curve_id,225:235] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,213:223,2] = 1
extended_class_maps[curve_id,213:223,-2] = 1
extended_class_maps[curve_id,225:235,2] = 1
extended_class_maps[curve_id,225:235,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 828
plotSpikesMap(curve_id) # 2 pics: Gk puis cl kappa
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,215:231] = 1
class_maps[curve_id,237:246] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,215:231,0] = 1
extended_class_maps[curve_id,215:231,-2] = 1
# extended_class_maps[curve_id,237:246,0] = 1
extended_class_maps[curve_id,237:246,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1174
plotSpikesMap(curve_id) # 1 pic : Mk au lieu de Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,222:229] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,222:229,2] = 1
extended_class_maps[curve_id,222:229,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1256
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,213:222] = 1
class_maps[curve_id,226:233] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,213:222,0] = 1
extended_class_maps[curve_id,213:222,-2] = 1
extended_class_maps[curve_id,226:233,0] = 1
extended_class_maps[curve_id,226:233,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1263
plotSpikesMap(curve_id) # 2 pics: Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,222:231] = 1
class_maps[curve_id,231:237] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,222:231,2] = 1
extended_class_maps[curve_id,222:231,-1] = 1
extended_class_maps[curve_id,231:237,2] = 1
extended_class_maps[curve_id,231:237,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1286
plotSpikesMap(curve_id) # 3 pics: Ml puis Gl, Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,216:223] = 1
class_maps[curve_id,226:248] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,216:223,2] = 1
extended_class_maps[curve_id,216:223,-1] = 1
extended_class_maps[curve_id,226:248,0] = 1
extended_class_maps[curve_id,226:248,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1363
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,213:221] = 1
class_maps[curve_id,235:244] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,213:221,0] = 1
extended_class_maps[curve_id,213:221,-2] = 1
extended_class_maps[curve_id,235:244,0] = 1
extended_class_maps[curve_id,235:244,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1531
plotSpikesMap(curve_id) # 2 pics: K puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,212:220] = 1
class_maps[curve_id,228:237] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
# extended_class_maps[curve_id,212:220,0] = 1
extended_class_maps[curve_id,212:220,-2] = 1
extended_class_maps[curve_id,228:237,0] = 1
extended_class_maps[curve_id,228:237,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1776
plotSpikesMap(curve_id) # G lambda (pas Gk)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,200:209] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,200:209,0] = 1
extended_class_maps[curve_id,200:209,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 153
plotSpikesMap(curve_id) # 2 pics: Mk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,219:228] = 1
class_maps[curve_id,230:239] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,219:228,2] = 1
extended_class_maps[curve_id,219:228,-2] = 1
extended_class_maps[curve_id,230:239,0] = 1
extended_class_maps[curve_id,230:239,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 324
plotSpikesMap(curve_id) # Gl puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,227:236] = 1
class_maps[curve_id,242:247] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,227:236,0] = 1
extended_class_maps[curve_id,227:236,-1] = 1
extended_class_maps[curve_id,242:247,0] = 1
extended_class_maps[curve_id,242:247,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 352
plotSpikesMap(curve_id) # 2 pics: Mk puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,222:231] = 1
class_maps[curve_id,244:250] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,222:231,2] = 1
extended_class_maps[curve_id,222:231,-2] = 1
extended_class_maps[curve_id,244:250,0] = 1
extended_class_maps[curve_id,244:250,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 434
plotSpikesMap(curve_id) # 2 pics: Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,218:224] = 1
class_maps[curve_id,226:235] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,218:224,0] = 1
extended_class_maps[curve_id,218:224,-2] = 1
extended_class_maps[curve_id,226:235,0] = 1
extended_class_maps[curve_id,226:235,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 920
plotSpikesMap(curve_id) # 2 pics: Mk les deux
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,209:227] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,209:227,2] = 1
extended_class_maps[curve_id,209:227,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1245
plotSpikesMap(curve_id) # Ml puis Mk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,202:214] = 1
class_maps[curve_id,232:237] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,202:214,2] = 1
extended_class_maps[curve_id,202:214,-1] = 1
extended_class_maps[curve_id,232:237,2] = 1
extended_class_maps[curve_id,232:237,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1408
plotSpikesMap(curve_id) # Ml puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,187:195] = 1
class_maps[curve_id,225:234] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,187:195,2] = 1
extended_class_maps[curve_id,187:195,-1] = 1
extended_class_maps[curve_id,225:234,0] = 1
extended_class_maps[curve_id,225:234,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1438
plotSpikesMap(curve_id) # que Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,214:229] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,214:229,0] = 1
extended_class_maps[curve_id,214:229,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1488
plotSpikesMap(curve_id) # 2 pics: Ak puis Gk
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,173:184] = 1
class_maps[curve_id,237:246] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,173:184,1] = 1
extended_class_maps[curve_id,173:184,-2] = 1
extended_class_maps[curve_id,237:246,0] = 1
extended_class_maps[curve_id,237:246,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1613
plotSpikesMap(curve_id) # que Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,226:240] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,226:240,0] = 1
extended_class_maps[curve_id,226:240,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

curve_id = 1711
plotSpikesMap(curve_id) # Gk puis Gl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,220:229] = 1
class_maps[curve_id,235:241] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,220:229,0] = 1
extended_class_maps[curve_id,220:229,-2] = 1
extended_class_maps[curve_id,235:241,0] = 1
extended_class_maps[curve_id,235:241,-1] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

# %%

curve_id = 497
plotSpikesMap(curve_id) # 2 pics: Gl puis Gk (initialement uniquement Gk annotée par Le Mans)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,222:230] = 1
class_maps[curve_id,242:255] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,222:230,0] = 1
extended_class_maps[curve_id,222:230,-1] = 1
extended_class_maps[curve_id,242:255,0] = 1
extended_class_maps[curve_id,242:255,-2] = 1
plotExtendedIFMaps(curve_id)
treated_samples.append(curve_id)

# Annotations à retirer (pas de pic(s))
curve_ids = [13, 140, 360, 576, 581, 615, 664, 698, 733, 821, 827, 858, 983, 1036, 1081, 1105, 1139, 1163, 1350, 1403, 1415, 1565, 1694, 1713, 1765, 1783, 1791]
for curve_id in curve_ids:
    extended_class_maps[curve_id,:,:] = 0
    # plotExtendedIFMaps(curve_id)

# %%

# QC check

# puis on check à la fin qu'on ait bien traité toutes les courbes
len(treated_samples) == len(untreated_samples)

np.setdiff1d(untreated_samples, treated_samples)
np.array([ 858, 1105, 1163, 1350, 1415]) # on a retiré ces pics juste avant, ok.

# %%

# Autres erreurs à corriger manuellement:
curve_id = 574
plotSpikesMap(curve_id) # IgMk et pas IgMl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,227:241] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,227:241,2] = 1
extended_class_maps[curve_id,227:241,-2] = 1
plotExtendedIFMaps(curve_id)

curve_id = 691
plotSpikesMap(curve_id)
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,246:252] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,246:252,0] = 1
extended_class_maps[curve_id,246:252,-1] = 1
plotExtendedIFMaps(curve_id)

curve_id = 1432
plotSpikesMap(curve_id) # Ig Gl et pas IgAl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,229:242] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,229:242,0] = 1
extended_class_maps[curve_id,229:242,-1] = 1
plotExtendedIFMaps(curve_id)

curve_id = 1182
plotSpikesMap(curve_id) # Ig Gk et pas IgGl
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,214:230] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,214:230,0] = 1
extended_class_maps[curve_id,214:230,-2] = 1
plotExtendedIFMaps(curve_id)

curve_id = 1786
plotSpikesMap(curve_id) # juste la Mk à la fin plutôt
np.where(class_maps[curve_id,:]==1)
class_maps[curve_id,:] = 0
class_maps[curve_id,228:240] = 1
plotSpikesMap(curve_id)
extended_class_maps[curve_id,:,:] = 0
extended_class_maps[curve_id,228:240,2] = 1
extended_class_maps[curve_id,228:240,-2] = 1
plotExtendedIFMaps(curve_id)

# %%

# Conversion en datasets pour l'analyse CNN:
if_x = np.stack([x,xG,xA,xM,xK,xL], axis=-1)
if_y = extended_class_maps

np.save(os.path.join(r'C:\Users\admin\Documents\Capillarys\data\2021\ifs','if_v1_x.npy'), if_x)
np.save(os.path.join(r'C:\Users\admin\Documents\Capillarys\data\2021\ifs','if_v1_y.npy'), if_y)


# %%

# Méthode d'analyse :
    
# 2-dim CNN ?
# 1-dim multi-channel CNN ?
# Multiple 1-dim 1-channel CNNs then concat ?
# Raw curves or deltas with ref ?
# Augmentation w/ GAN ?

































