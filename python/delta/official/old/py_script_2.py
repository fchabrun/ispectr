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

# since the callbacks will handle adapting the learning rate and stopping training when no performance improvement is observed after a few epochs
PATH_IN = r"D:\Anaconda datasets\Capillarys\IF_transformer" # path for data (in)
PATH_OUT = r"D:\Anaconda datasets\Capillarys\IF_transformer\output" # path for output (out)
normc = False

DUMMY_DATASET=False # set this to true if you with to use the dummy 4-samples dataset (for debugging purposes only)

# %%

##########################################################
##########################################################
#####                   BASE PART                    #####
##########################################################
##########################################################

# please make sure you have the library "Xlsxwriter" installed for exporting the "preddy" contingency table

import time
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from matplotlib import pyplot as plt
from scipy.stats import norm
from colour import Color

##############################
# LOAD AND PREPROCESS DATA ###
##############################

# LOAD DATA
if DUMMY_DATASET:
    if_x = np.load(os.path.join(PATH_IN,'NPY_file_X.npy'))
    if_y = np.load(os.path.join(PATH_IN,'NPY_file_Y.npy'))
else:
    if_x = np.load(os.path.join(PATH_IN,'if_v1_x.npy'))
    if_y = np.load(os.path.join(PATH_IN,'if_v1_y.npy'))

# PARTITIONING
# first we should partition our data
# since in this example we only have 2 samples, we will put those same samples in the training, validation and test sets
# however, we should have 3 datasets:
# 1) training -> data used for training models
# 2) validation -> data used for monitoring loss and metrics during training, stop training when performance doesn't improve and compare architectures when changing hyperparameters
# 3) test -> for testing the performance of the final model on new, unseen data

# for this dummy example, just copy arrays for training, validation and test
if DUMMY_DATASET:
    x_train = if_x
    y_train = if_y
    x_valid = if_x
    y_valid = if_y
    x_test = if_x
    y_test = if_y
else:
    # if using the real dataset, partition first
    rng = np.random.RandomState(seed=1)
    train_samples = rng.choice(np.arange(if_x.shape[0]), size=if_x.shape[0]//2, replace=False)
    valid_samples = rng.choice(np.setdiff1d(np.arange(if_x.shape[0]), train_samples), size=(if_x.shape[0]-train_samples.shape[0])//2, replace=False)
    test_samples = np.setdiff1d(np.arange(if_x.shape[0]), np.concatenate([train_samples,valid_samples]))
    
    x_train = if_x[train_samples,...]
    x_valid = if_x[valid_samples,...]
    x_test = if_x[test_samples,...]
    y_train = if_y[train_samples,...]
    y_valid = if_y[valid_samples,...]
    y_test = if_y[test_samples,...]

spe_width = 304 # define the size of the input (i.e. 304 points)

# STANDARDIZATION
# we could standardize data in order to achieve 0 mean and 1 standard deviation for all points of curves
# however, by experience we deducted that does not grant significant improvement of results
if normc:
    # compute & apply mean & std
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    # replace 0's
    x_std[x_std==0] = 1
    x_train = (x_train-x_mean)/x_std
    x_valid = (x_valid-x_mean)/x_std
    x_test = (x_test-x_mean)/x_std

print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_train.shape))
print('validation set X shape: '+str(x_valid.shape))
print('validation set Y shape: '+str(y_valid.shape))
print('supervision set X shape: '+str(x_test.shape))
print('supervision set Y shape: '+str(y_test.shape))

# %%

##########################################################
##########################################################
#####                 TRAINING PART                  #####
##########################################################
##########################################################

BASE_LR = 1e-3 # start training with learning rate at 1E-3
MIN_LR = 1e-5 # do not reduce learning rate below 1E-5
BATCH_SIZE = 32 #  default batch size, increase in order to speed up training, decrease in order to potentially increase performance
N_EPOCHS = 1000 # please note that the number of epochs is usually irrelevant and only set to a high value,
MODEL_NAME = "final_model_basic.h5" # model name, for saving (during training) or loading (during inference)

#############################
#  CREATE SAMPLE GENERATOR  #
#############################

# create generator for augmentation
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=True, add_fake_mspikes=None, add_fake_flcs=None, add_noise=None, add_shifting=None, seed=42, output='parallel', full_batches_only=True):
        self.x                  = x             # the x array
        self.y                  = y             # the y array
        self.batch_size         = batch_size    # the batch size
        self.shuffle            = shuffle       # set to true for shuffling samples
        self.seed               = seed          
        self.add_fake_mspikes   = add_fake_mspikes
        self.add_fake_flcs      = add_fake_flcs
        self.add_noise          = add_noise
        self.add_shifting       = add_shifting
        self.rng                = np.random.RandomState(self.seed)
        self.output             = output # according to model architecture (input & output)
        self.full_batches_only  = full_batches_only
        self.on_epoch_end()

    def __len__(self): # should return the number of batches per epoch
        if self.full_batches_only:
            return self.x.shape[0]//self.batch_size
        return int(np.ceil(self.x.shape[0]/self.batch_size))

    def on_epoch_end(self): # shuffle samples
        self.indexes = np.arange(self.x.shape[0])
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        # get indexes for current batch
        batch_indexes = self.indexes[index*self.batch_size:((index+1)*self.batch_size)]
        # get data
        batch_x = self.x[batch_indexes,...].copy()
        batch_y = self.y[batch_indexes,...].copy()
        # augment id needed
        if self.add_noise is not None:
            freq=self.add_noise['freq'] # between 0. and 1., how often do wee add noise
            for i in range(batch_x.shape[0]):
                if self.rng.random(1) < freq:
                    std=self.rng.uniform(low=self.add_noise['minstd'], high=self.add_noise['maxstd'], size=1)
                    batch_x[i,...] += self.rng.normal(loc=0, scale=std, size=batch_x.shape[1:])
        if self.add_shifting is not None:
            freq=self.add_shifting['freq'] # between 0. and 1., how often do wee add noise
            for i in range(batch_x.shape[0]):
                if self.rng.random(1) < freq:
                    d=self.rng.choice(np.arange(self.add_shifting['min'],self.add_shifting['max']))
                    if self.rng.random() > .5: # left
                        batch_x[i,...] = np.concatenate([batch_x[i,d:,...],np.zeros_like(batch_x[i,:d,...])], axis=0)
                        batch_y[i,...] = np.concatenate([batch_y[i,d:,...],np.zeros_like(batch_y[i,:d,...])], axis=0)
                    else: # right
                        batch_x[i,...] = np.concatenate([np.zeros_like(batch_x[i,-d:,...]),batch_x[i,:-d,...]], axis=0)
                        batch_y[i,...] = np.concatenate([np.zeros_like(batch_y[i,-d:,...]),batch_y[i,:-d,...]], axis=0)
        if self.add_fake_mspikes is not None:
            freq=self.add_fake_mspikes['freq'] # between 0. and 1., how often do wee add noise
            for i in range(batch_x.shape[0]):
                if self.rng.random(1) < freq:
                    # add a fake mspike
                    for ms in range(self.rng.choice(np.arange(self.add_fake_mspikes['mincount'],self.add_fake_mspikes['maxcount']+1))):
                        # choose random dims
                        heavy_dim = self.rng.choice([1,2,3])
                        light_dim = self.rng.choice([4,5])
                        pos=self.rng.choice(np.arange(self.add_fake_mspikes['minpos'],self.add_fake_mspikes['maxpos']))
                        height=self.rng.uniform(self.add_fake_mspikes['minheight'],self.add_fake_mspikes['maxheight'])
                        width=self.rng.uniform(self.add_fake_mspikes['minwidth'],self.add_fake_mspikes['maxwidth'])
                        curve_with_mspike = norm.pdf(np.arange(304),pos,width)*height
                        for dim in range(batch_x.shape[-1]):
                            if (dim==heavy_dim) | (dim==light_dim): # do not add! but add to y
                                batch_y[i,...,dim-1] = np.maximum(batch_y[i,...,dim-1], ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1)
                            else:
                                batch_x[i,...,dim] += curve_with_mspike + curve_with_mspike*np.random.normal(0,.01,curve_with_mspike.shape)
        if self.add_fake_flcs is not None:
            freq=self.add_fake_flcs['freq'] # between 0. and 1., how often do wee add noise
            for i in range(batch_x.shape[0]):
                if self.rng.random(1) < freq:
                    # add a fake mspike
                    for ms in range(self.rng.choice(np.arange(self.add_fake_flcs['mincount'],self.add_fake_flcs['maxcount']+1))):
                        # choose random dims
                        light_dim = self.rng.choice([4,5])
                        pos=self.rng.choice(np.arange(self.add_fake_flcs['minpos'],self.add_fake_flcs['maxpos']))
                        height=self.rng.uniform(self.add_fake_flcs['minheight'],self.add_fake_flcs['maxheight'])
                        width=self.rng.uniform(self.add_fake_flcs['minwidth'],self.add_fake_flcs['maxwidth'])
                        curve_with_mspike = norm.pdf(np.arange(304),pos,width)*height
                        for dim in range(batch_x.shape[-1]):
                            if dim==light_dim: # do not add! but add to y
                                batch_y[i,...,dim-1] = np.maximum(batch_y[i,...,dim-1], ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1)
                            else:
                                batch_x[i,...,dim] += curve_with_mspike + curve_with_mspike*np.random.normal(0,.01,curve_with_mspike.shape)
        
        if self.output is not None:
            if self.output == 'parallel':
                # invert y
                # pour chaque point de chaque courbe :
                # si courbe ref (i.e max toutes) == 0
                # -> pas de pic, laisser à 0
                # si courbe ref == 1 (un pic)
                # si 0 -> 1, sinon 0
                # globalement on inverse, donc chaîne sans pic devient full 1 et chaîne avec pic devient full 1 sauf 0 sur les pic
                # puis on fait * coure de ref -> chaîne sans pic devient full 0 et 1 aux pics
                # et chaîne aec pic devient 0 partout car 0*1 (réf*courbe) et 1*0 (réf*courbe)
                batch_y_ref = batch_y.max(axis=-1)
                for dim in range(batch_y.shape[-1]):
                    batch_y[...,dim] = (1-batch_y[...,dim]) * batch_y_ref
                # puis mettre la dimension -1 dans la 1e dimension
                batch_x = np.concatenate([batch_x[...,dim] for dim in range(1,batch_x.shape[-1])], axis=0)
                batch_y = np.concatenate([batch_y[...,dim] for dim in range(batch_y.shape[-1])], axis=0)
                return np.expand_dims(batch_x, (2,3)), np.expand_dims(batch_y, (2,3))
                
        return batch_x, batch_y
    
# create our train generator
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=BATCH_SIZE,
                        # add_fake_mspikes=None,
                        add_fake_mspikes=dict(freq=.5,
                                              minpos=180, # 180 is nice
                                              maxpos=251, # 251 is nice
                                              minheight=.5, # .5 is nice
                                              maxheight=8, # 8 is nice
                                              minwidth=3.5, # 3.5 is nice
                                              maxwidth=4.5, # 4.5 is nice
                                              mincount=1,
                                              maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                        # add_fake_flcs=None,
                        add_fake_flcs=dict(freq=.1,
                                            minpos=100, # 180 is nice
                                            maxpos=251, # 251 is nice
                                            minheight=.4, # .5 is nice
                                            maxheight=.5, # 8 is nice
                                            minwidth=2.5, # 3.5 is nice
                                            maxwidth=3.5, # 4.5 is nice
                                            mincount=1,
                                            maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                        # add_noise=None,
                        add_noise=dict(freq=.2, minstd=.001, maxstd=.002),
                        # add_shifting=None,
                        add_shifting=dict(freq=.8, min=1, max=4),
                        shuffle = True,
                        seed = 0,
                        output = 'parallel')

validaug_gen = ITGenerator(x=x_valid, y=y_valid, batch_size=BATCH_SIZE,
                           # add_fake_mspikes=None,
                           add_fake_mspikes=dict(freq=.5,
                                                 minpos=180, # 180 is nice
                                                 maxpos=251, # 251 is nice
                                                 minheight=.5, # .5 is nice
                                                 maxheight=8, # 8 is nice
                                                 minwidth=3.5, # 3.5 is nice
                                                 maxwidth=4.5, # 4.5 is nice
                                                 mincount=1,
                                                 maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                           # add_fake_flcs=None,
                           add_fake_flcs=dict(freq=.1,
                                              minpos=100, # 180 is nice
                                              maxpos=251, # 251 is nice
                                              minheight=.4, # .5 is nice
                                              maxheight=.5, # 8 is nice
                                              minwidth=2.5, # 3.5 is nice
                                              maxwidth=3.5, # 4.5 is nice
                                              mincount=1,
                                              maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                           # add_noise=None,
                           add_noise=dict(freq=.2, minstd=.001, maxstd=.002),
                           # add_shifting=None,
                           add_shifting=dict(freq=.8, min=1, max=4),
                           shuffle = False,
                           seed = 0,
                           output = 'parallel')

if False:
    # visualize generator results
    
    # a function for making plots for debugging our augmentation function
    def plotDebug(xnoaug, ynoaug, xwithaug, ywithaug):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10,9))
        for i,x,y in [[0,xnoaug,ynoaug],[1,xwithaug,ywithaug]]:
            for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
                if (i==1) & (num==0):
                    continue
                plt.subplot(6,2,i+1+num*2)
                plt.plot(np.arange(0,spe_width), x[:,num], '-', color = col, label = lab)
                if num==0:
                    class_map=y.max(axis=-1)
                else:
                    class_map=y[...,num-1]
                for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                    plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=x[peak_start:peak_end,num], color="red")
        plt.tight_layout()
    
    # plot debug
    batch_ind = 0 # which batch to compute
    
    # run aug + get unaugmented curves
    batch_x, batch_y = train_gen.__getitem__(batch_ind)
    batch_x = batch_x[...,0,0]
    batch_y = batch_y[...,0,0]
    noaug_batch_x = train_gen.x[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size]]
    noaug_batch_y = train_gen.y[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size]]
    # reshape
    # [[b+i*FLAGS.batch_size for i in range(5)] for b in range(FLAGS.batch_size)]
    batch_x = np.transpose(np.stack([batch_x[[b+i*BATCH_SIZE for i in range(5)],...] for b in range(BATCH_SIZE)], axis=-1), axes=(2,1,0))
    batch_y = np.transpose(np.stack([batch_y[[b+i*BATCH_SIZE for i in range(5)],...] for b in range(BATCH_SIZE)], axis=-1), axes=(2,1,0))
    # add fake ref curve for generated x
    batch_x = np.concatenate([np.expand_dims(batch_x.max(axis=-1), -1), batch_x], -1)
    # plot
    # for sample_ind in range(1): # which sample to plot
    for sample_ind in range(2): # which sample to plot
        plotDebug(noaug_batch_x[sample_ind], noaug_batch_y[sample_ind], batch_x[sample_ind], batch_y[sample_ind])
        

#############################
# CREATE MODEL FOR TRAINING #
#############################

# BUILD MODEL (1D U-NET) BACKBONE

# define a u-net block
def conv1d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

# define a u-net model (based on u-net blocks)
def get_unet(input_signal, n_filters=16, dropout=0.5, batchnorm=True, n_classes=5, contract=1, return_model=True, return_sigmoid=True):
    # contracting path
    c1 = conv1d_block(input_signal, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2,1)) (c1)
    p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

    c2 = conv1d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.layers.MaxPooling2D((2,1)) (c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = conv1d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.layers.MaxPooling2D((2,1)) (c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = conv1d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = tf.keras.layers.MaxPooling2D((2,1)) (c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)
    
    c5 = conv1d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = tf.keras.layers.Conv2DTranspose(n_filters*8, (3,1), strides=(2,1), padding='same') (c5)
    u6 = tf.keras.layers.Concatenate() ([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = conv1d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3,1), strides=(2,1), padding='same') (c6)
    u7 = tf.keras.layers.Concatenate() ([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = conv1d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3,1), strides=(2,1), padding='same') (c7)
    u8 = tf.keras.layers.Concatenate() ([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = conv1d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3,1), strides=(2,1), padding='same') (c8)
    u9 = tf.keras.layers.Concatenate() ([u9, c1])
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = conv1d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = tf.keras.layers.Conv2D(n_classes, (1,contract), activation='sigmoid') (c9)
    if return_model:
        model = tf.keras.models.Model(inputs=[input_signal], outputs=[outputs])
        return model
    if return_sigmoid:
        return outputs
    else:
        return c9
    
# CONSTRUCT, INSTANCIATE AND COMPILE MODEL

input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_is')
model = get_unet(input_signal, n_filters=32, dropout=0.05, batchnorm=True, n_classes=1)

# we define the callbacks that will be used for monitoring training
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=1),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=MIN_LR, verbose=1),
             tf.keras.callbacks.ModelCheckpoint(os.path.join(PATH_OUT,MODEL_NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
             ]

# and we then can compile the model for training
model.compile(loss="binary_crossentropy", # default loss function as described in the u-net paper
              optimizer=tf.keras.optimizers.Adam(lr=BASE_LR)) # chosen base learning rate

print(model.summary())

############
# TRAINING #
############

# finally, once our input data are reshaped, we can start training
results = model.fit(train_gen, # set training x and y
                    batch_size=BATCH_SIZE, # batch size
                    epochs=N_EPOCHS, # number of epochs
                    callbacks=callbacks, # callbacks defined earlier
                    verbose=2, # set to verbose = 1 for further details during training (i.e. per batch instead of per epoch)
                    validation_data=validaug_gen, # validation x and y
                    validation_steps = validaug_gen.__len__())

# save history
with open(os.path.join(PATH_OUT,"training_log.pkl"), 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%

##########################################################
##########################################################
#####                 INFERENCE PART                 #####
##########################################################
##########################################################

<<<<<<< HEAD
#Xavier : mets
Deep_sup = True
INPUT_DIMENSIONS = 3
POST_PROCESS_METHOD = ('none','invert_predictions')[0]
POST_PROCESS_SQUARE = False
POST_PROCESS_SMOOTH = False
MODEL_NAME = "model_SiT4+.h5"

=======
>>>>>>> 914a8fda1f87eeb7f25198ba8b3aa8837e0e2d62
# ces infos sont à prendre en compte uniquement si on change l'architecture du modèle
#INPUT_DIMENSIONS = 2 # 2 for basic unet, 3 for any model accepting 3D input
#POST_PROCESS_METHOD = ('none','invert_predictions')[1]
#POST_PROCESS_SQUARE = True # allows reducing the artifactual ghosting effect before and after mspikes, in invert_predictions mode
#POST_PROCESS_SMOOTH = False # allows reducing the artifactual ghosting effect before and after mspikes, in invert_predictions mode
#MODEL_NAME = "" # model name, for saving (during training) or loading (during inference)

##################################
#           INFERENCE            #    
##################################


# class to implement the segformer transformer building block (w/o efficient attention)
class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, key_dim, ff_dim, rate=0., seed=42, **kwargs):
        super(Transformer_Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.seed = seed
        self.key_dim = key_dim
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                      key_dim=self.key_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.ff_dim,
                                   kernel_initializer="he_normal"),
             tf.keras.layers.Conv1D(self.ff_dim, 3, kernel_initializer='he_normal',
                                    padding='same'),
             tf.keras.layers.Activation("gelu"),
             tf.keras.layers.Dense(self.embed_dim, 
                                   kernel_initializer="he_normal")]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate, seed=self.seed)
        self.dropout2 = tf.keras.layers.Dropout(self.rate, seed=self.seed)

    def call(self, inputs, training):
        inputs_n = self.layernorm1(inputs)
        attn_output = self.att(inputs_n, inputs_n)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm2(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
    
            config = super().get_config().copy()
            config.update({
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'ff_dim': self.ff_dim,
            })
            return config


# on charge le modèle voulu (le nom du modèle est défini dans la première section)
model = tf.keras.models.load_model(os.path.join(PATH_IN, MODEL_NAME), 
                                   compile=False,
                                   custom_objects={'Transformer_Block': Transformer_Block})


size = x_test.shape[0] # determine size of the dataset

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


start=time.time() # register start and end times in order to compute speed
if Deep_sup :
    y_test_raw_=model.predict(test_input)[0] # actual inference
    #y_test_raw_ = np.mean(y_test_raw_[0:1], axis=0)
else :
    y_test_raw_=model.predict(test_input)
end=time.time()

print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 7.191 us per sample
print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.7s per 100 samples

##################################
#  POST PROCESS MODEL'S OUTPUT   #    
##################################

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
            for d in range(y_test.shape[-1]):
                y_test_[i,...,d] = smooth_curve(y_test_[i,...,d])
                
##################################
#     PLOTS (FOR DEBUGGING)      #    
##################################

# Simple plot function : raw, but efficient
def debugPlotSample(ix=0):
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
        ax_dict["{} - ground truth".format(dim_name)].plot(np.arange(304), y_test[ix,:,d])
        ax_dict["{} - ground truth".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - ground truth".format(dim_name)].text(0,1,"{} - ground truth".format(dim_name), verticalalignment="top")
        # third column : plot the raw predictions
        ax_dict["{} - raw preds".format(dim_name)].plot(np.arange(304), y_test_raw_[ix*5+d,:,0,0])
        ax_dict["{} - raw preds".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - raw preds".format(dim_name)].text(0,1,"{} - raw preds".format(dim_name), verticalalignment="top")
        # fourth column : plot the processed predictions
        ax_dict["{} - proc preds".format(dim_name)].plot(np.arange(304), y_test_[ix,:,d])
        ax_dict["{} - proc preds".format(dim_name)].set_ylim((0,1))
        ax_dict["{} - proc preds".format(dim_name)].text(0,1,"{} - proc preds".format(dim_name), verticalalignment="top")
    plt.tight_layout()
        
debugPlotSample(0) # simple visual test to check that the model outputs consistent predictions
# and that our post-processing algorithms works OK

# More refined function : prettier, but less information displayed
def plotITPredictions(ix):
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

# cette métrique ci-dessous est peu intéressante car elle concerne le % correctement classé de TOUS les points
# la majorité sera =0 et probablement correctement classée =0, donc elle devrait tourner autour de 99%+
print("Mean per-point accuracy (threshold={}):".format(threshold))
for k in range(pr.shape[1]):
    print("    > Fraction '{}': {:.1f}% +- {:.1f}".format(['G','A','M','k','l'][k],100*np.nanmean(pr[:,k]),np.nanstd(pr[:,k])))


# on retire la partie suivante, "par pic", qui n'est pas celle qui nous intéresse
# puisqu'on souhaite plutôt travailler sur les métriques "par courbe"

# below we compute the confusion matrix
# i.e. we parse all ground truth m-spikes
# for each m-spike, we observe the median prediction of the neural network at this location
# and we build a confusion matrix according the the type of m-spike (ground truth) and the prediction
# i.e. any values higher than threshold (if not -> no m-spike detected) and median type (i.e. g, a, m, k, l)

# threshold=.1
# curve_ids = []
# groundtruth_spikes = []
# predicted_spikes = []
# for ix in trange(x_test.shape[0]):
#     flat_gt = np.zeros_like(y_test[ix,:,0])
#     for i in range(y_test.shape[-1]):
#         flat_gt += y_test[ix,:,i]*(1+np.power(2,i))
#     gt_starts = []
#     gt_ends = []
#     prev_v = 0
#     for i in range(spe_width):
#         if flat_gt[i] != prev_v: # changed
#             # multiple cases:
#             # 0 -> non-zero = enter peak
#             if prev_v == 0:
#                 gt_starts.append(i)
#             # non-zero -> 0 = out of peak
#             elif flat_gt[i] == 0:
#                 gt_ends.append(i)
#             # non-zero -> different non-zero = enter other peak
#             else:
#                 gt_ends.append(i)
#                 gt_starts.append(i)
#             prev_v = flat_gt[i]
            
#     if len(gt_starts) != len(gt_ends):
#         raise Exception('Inconsistent start/end points')
    
#     if len(gt_starts)>0:
#         # for each m-spike, we detect what Ig type the model predicted at this location
#         for pstart,pend in zip(gt_starts,gt_ends):
#             gt_ig_denom = ''
#             if np.sum(y_test[ix,pstart:pend,:3])>0:
#                 HC_gt = int(np.median(np.argmax(y_test[ix,pstart:pend,:3], axis=1)))
#                 gt_ig_denom = ['G','A','M'][HC_gt]
#             lC_gt = int(np.median(np.argmax(y_test[ix,pstart:pend,3:], axis=1)))
#             gt_ig_denom += ['k','l'][lC_gt]
            
#             pred_ig_denom = ''
#             if np.sum(y_test_[ix,pstart:pend,0,:]>threshold)>0: # un pic a été détecté
#                 if np.sum(y_test_[ix,pstart:pend,0,:3]>threshold)>0:
#                     HC_pred = int(np.median(np.argmax(y_test_[ix,pstart:pend,0,:3], axis=1)))
#                     pred_ig_denom = ['G','A','M'][HC_pred]
#                 lC_pred = int(np.median(np.argmax(y_test_[ix,pstart:pend,0,3:], axis=1)))
#                 pred_ig_denom += ['k','l'][lC_pred]
#             else:
#                 pred_ig_denom = 'none'
                
#             groundtruth_spikes.append(gt_ig_denom)
#             predicted_spikes.append(pred_ig_denom)
#             curve_ids.append(ix)
#     else:
#         pass

# # we summarize previously computed data into a data.frame used for building the confusion matrix
# conc_df = pd.DataFrame(dict(ix=curve_ids,
#                             true=groundtruth_spikes,
#                             pred=predicted_spikes))

# # and finally print the confusion matrix
# print(pd.crosstab(conc_df.true, conc_df.pred))

# # finally, we use these same data in order to compute global accuracy and accuracy for each type of m-spike
# print('Global accuracy: '+str(round(100*np.sum(conc_df.true==conc_df.pred)/conc_df.shape[0],1)))
# for typ in np.unique(conc_df.true):
#     subset=conc_df.true==typ
#     print('  Accuracy for type '+typ+': '+str(round(100*np.sum(conc_df.true.loc[subset]==conc_df.pred.loc[subset])/np.sum(subset), 1)))
    
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

writer = pd.ExcelWriter(os.path.join(PATH_OUT,"Accuracy table.xlsx"), engine='xlsxwriter')
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


    
np.where((sample_gt_classes == 'Normal') & (sample_pred_classes == 'Ig Gl'))  
    
debugPlotSample(335)
    

    
    
    
    

    
    
    
    

    
    
    
    
    
    
    