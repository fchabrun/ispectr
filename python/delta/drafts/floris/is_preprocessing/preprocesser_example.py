# -*- coding: utf-8 -*-
"""
Created on 07/08/21

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

import time
import numpy as np
import tensorflow as tf
import os

# On définit ci-dessous le chemin vers le modèle et le nom du modèle
model_path = r"C:\Users\admin\Documents\Capillarys\reports\DELTA\Synthèse\AddendumJul21"
model_name = 'is_preprocesser_v1.h5'

# On définit également le chemin vers les matrices de données
data_path = "C:/Users/admin/Documents/Capillarys/data/2021/ifs"

# on charge les données
if_x = np.load(os.path.join(data_path,'if_v1_x.npy'))

if_x.shape
# (1803, 304, 6)
# cette matrice "if_x" contient 1803 échantillons, chacun composé de 6 courbes de 304 points: ELP, G, A, M, k, l
# on n'a besoin pour "localiser" les pics que de la courbe de référence (ELP) : la première
# on va donc se débarrasser des autres
if_x = if_x[...,0]
if_x.shape # (1803, 304)

# on charge le modèle
model = tf.keras.models.load_model(filepath = os.path.join(model_path,model_name), compile=False)
# on n'a pas besoin de compiler le modèle car on ne va l'utiliser que pour l'inférence des nouveaux échantillons

size = if_x.shape[0]
start=time.time() # on note le temps de départ
y_raw_ = model.predict(np.expand_dims(if_x, axis=(-1,-2))) # on réalise l'inférence ppmt dit
y_ = y_raw_[...,0,0] # pas besoin de garder les 2 dernières dimensions qui servaient uniquement aux convolutions
end=time.time() # et le temps de fin de la prédiction

# on peut déterminer le temps passé à prédire:
print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 1.583 ms per sample
print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.2s per 100 samples

# enfin, on va post-processer l'output du modèle
# en binarisant les prédictions, i.e., prédiction>=0.5 -> 1, prédiction<0.5 -> 0
y_binary_ = (y_>.5)*1.

# on peut alors vérifier visuellement si le modèle a l'air d'avoir correctement fonctionné:
    
def plotELPPred(ix):
    # on importe matplotlib pour réaliser la figure
    from matplotlib import pyplot as plt 
    # on crée une nouvelle figure
    plt.figure(figsize=(12,6))
    # on crée une sous figure pour l'input (x)
    plt.subplot(3,1,1)
    plt.plot(np.arange(0,304), if_x[ix,...])
    plt.title("Input curve")
    # puis la 2e sous figure pour le y prédit (y_)
    plt.subplot(3,1,2)
    plt.plot(np.arange(0,304), y_[ix,...])
    plt.title("Prediction")
    plt.tight_layout()
    # et enfin la 3e pour la prédiction "binarisée"
    plt.subplot(3,1,3)
    plt.plot(np.arange(0,304), y_binary_[ix,...])
    plt.title("Prediction (binary)")
    plt.tight_layout()

# pour l'échantillon numéro 0:    
plotELPPred(0)

# puisque tout a l'air d'avoir correctement fonctionné, on peut sauvegarder nos prédictions (binarisées)
# il ne restera plus qu'à vérifier à la main chaque prédiction & convertir cette matrice
# afin de déterminer le type d'Ig pour chaque échantillon
# car ici on a simplement "placé" le pic, mais on ne l'a pas typé

    



    



    



