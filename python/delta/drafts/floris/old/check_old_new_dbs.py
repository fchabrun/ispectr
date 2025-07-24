# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:40:02 2021

@author: admin
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

initial_db = pd.read_csv(r"C:\Users\admin\Documents\Capillarys\data\2021\ifs\lemans_if_matv1.csv")

new_db_x = np.load(r'C:\Users\admin\Documents\Capillarys\data\2021\ifs\if_v1_x.npy')
new_db_y = np.load(r'C:\Users\admin\Documents\Capillarys\data\2021\ifs\if_v1_y.npy')

# la fonction pour convertir de new en old annotations
def classifySample(postprocessed_predicted_curves):
    """
    cette fonction prend en entrée une np.array de dimension 304x5 (les 5 courbes post processed prédites par le modèle)
    et renvoie une str: la classe de l'échantillon
    peut marcher à la fois pour les annotations réelles (ground truth) et les prédictions (si seuillées)
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

new_db_labels = []
for i in tqdm(range(new_db_x.shape[0])):
    # la classe réelle
    new_db_labels.append(classifySample(new_db_y[i,...]))

igs_sorted = ['iggk','iggl','igak','igal','igmk','igml']
initial_labels = pd.Series(np.argmax(np.array(initial_db.loc[:,igs_sorted]), axis=1)).map({i:v for i,v in enumerate(igs_sorted)}).tolist()

new_db_labels_pd = pd.Series(new_db_labels)
initial_labels_pd = pd.Series(initial_labels)

pd.crosstab(initial_labels_pd, new_db_labels_pd)

# col_0  Complex  Ig Ak  Ig Al  Ig Gk  Ig Gl  Ig Mk  Ig Ml  Normal
# row_0                                                           
# igak         1    102      0      0      0      1      0       3
# igal         5      0     70      0      1      0      0       1
# iggk        53      0      0    700      2      1      0      44
# iggl         8      0      0      1    378      1      0       4
# igmk         9      0      0      0      0    331      0       1
# igml         0      0      0      0      0      1     85       0

# il y a l'air d'avoir beaucoup de discordances ?!
# notamment beaucoup de "normal", ce qui n'était pas attendu
# regardons ces courbes


def plotITPredictions(ix):
    from matplotlib import pyplot as plt
    spe_width = 304
    
    old_names = ["ELPx","IgGx","IgAx","IgMx","Kx","Lx"]
    old_x_values = []
    for old_name in old_names:
        initial_values = np.array(initial_db.iloc[ix,:].loc[["{}{}".format(old_name,i) for i in range(1,305)]])
        initial_values /= np.max(initial_values)
        old_x_values.append(initial_values)
    old_x_values = np.stack(old_x_values, axis=-1)
    
    # les courbes ne sont pas identiques ?!
    # ix=0
    # initial_values = np.array(initial_db.iloc[ix,:].loc[["IgGx{}".format(i) for i in range(1,305)]])
    # initial_values /= np.max(initial_values)
    # np.sum(initial_values - new_db_x[ix,:,0])
    
    plt.figure(figsize=(14,8))
    plt.subplot(3,1,1)
    # on récupère la class map (binarisée)
    # class_map = y[ix].max(axis=1)
    curve_values = old_x_values
    for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width), curve_values[:,num], '-', color = col, label = lab)
    plt.title('Sample #{} old DB'.format(ix))
    plt.legend()
    
    plt.subplot(3,1,2)
    # on récupère la class map (binarisée)
    # class_map = y[ix].max(axis=1)
    curve_values = new_db_x[ix,:]
    for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width), curve_values[:,num], '-', color = col, label = lab)
    plt.title('Sample #{} new DB'.format(ix))
    plt.legend()
    # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
    #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

    # on plot aussi les autres courbes
    plt.subplot(3,1,3)
    for num,col,lab in zip(range(5), ['purple','pink','green','red','blue'], ['G','A','M','k','l']):
        plt.plot(np.arange(0,spe_width)+1, new_db_y[ix,:,num]/5+(4-num)/5, '-', color = col, label = lab)
    plt.ylim(-.05,1.05)
    plt.legend()
    plt.title('New ground truth maps // Old label = "{}"'.format(initial_labels[ix]))
    plt.tight_layout()

# now find discrepancies

np.where(np.logical_and(initial_labels_pd == "igak", new_db_labels_pd == "Complex"))
# this one is ~ ok
np.where(np.logical_and(initial_labels_pd == "igal", new_db_labels_pd == "Complex"))
# seems okay too

np.where(np.logical_and(initial_labels_pd == "iggk", new_db_labels_pd == "Complex"))
# not done yet
np.where(np.logical_and(initial_labels_pd == "iggl", new_db_labels_pd == "Complex"))
# not done yet
np.where(np.logical_and(initial_labels_pd == "igmk", new_db_labels_pd == "Complex"))
# not done yet

np.where(np.logical_and(initial_labels_pd == "iggl", new_db_labels_pd == "Ig Gk"))
# indeed was an error (100% sure)
np.where(np.logical_and(initial_labels_pd == "igal", new_db_labels_pd == "Ig Gl"))
# indeed was an error (100% sure)
np.where(np.logical_and(initial_labels_pd == "iggk", new_db_labels_pd == "Ig Gl"))
# was indeed errors (100% sure for first, 80% for second)

np.where(np.logical_and(initial_labels_pd == "igak", new_db_labels_pd == "Ig Mk"))
# pretty sure it was an error
np.where(np.logical_and(initial_labels_pd == "iggk", new_db_labels_pd == "Ig Mk"))
# not sure
np.where(np.logical_and(initial_labels_pd == "iggl", new_db_labels_pd == "Ig Mk"))
# pretty sure it was an error
np.where(np.logical_and(initial_labels_pd == "igml", new_db_labels_pd == "Ig Mk"))
# 100% sure it was an error

np.where(np.logical_and(initial_labels_pd == "iggk", new_db_labels_pd == "Normal"))
# indeed many curves labeled iggk were actually normal ! :(

for i in np.where(np.logical_and(initial_labels_pd == "iggk", new_db_labels_pd == "Normal"))[0]:
    plotITPredictions(i)

plotITPredictions(574)

# En conclusion on peut garder les annotations telles qu'on les a corrigées
# Et simplement retransformer ça en annotations originales

np.unique(new_db_labels, return_counts=True)

# étrange par contre -> pas de chaîne légère libre ??
# en effet peut être jamais seules ! (i.e. tjs sur une courbe avec au moins un pic entier)

new_db_labels_pd.to_csv(r"C:\Users\admin\Documents\Capillarys\data\2021\ifs\sebia_labeled\if_simple_y.csv", index=False, header=False)

new_db_x.shape





























