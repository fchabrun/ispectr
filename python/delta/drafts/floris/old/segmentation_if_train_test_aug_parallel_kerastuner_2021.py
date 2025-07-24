# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

print('Starting IF classification script...')

import argparse
import time
from tqdm import trange
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from scipy.stats import norm
import kerastuner as kt

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--step", type=str, default='train')
parser.add_argument("--loss", type=str, default='binary_crossentropy')
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
do_part = FLAGS.part==1

model_name = 'segifaug-{}-loss-{}.h5'.format('parallelunetv1',FLAGS.loss)
log_name = model_name[:-2]+"pkl"

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if debug_mode == 1:
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = 'C:/Users/admin/Documents/Capillarys/temp2021/kerastuner_unetparallel'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/out'
    
# debug_mode=1
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

# Partitionnement
if do_part:
    part_rng = np.random.RandomState(seed=42)
    
    train_part = part_rng.choice(a = np.arange(if_x.shape[0]), size = if_x.shape[0]//2, replace = False)
    test_part = part_rng.choice(a = np.setdiff1d(np.arange(if_x.shape[0]), train_part), size = (if_x.shape[0]-train_part.shape[0])//2, replace = False)
    valid_part = np.setdiff1d(np.setdiff1d(np.arange(if_x.shape[0]), train_part), test_part)
    
    np.save(os.path.join(path_out, 'train_part.npy'), train_part)
    np.save(os.path.join(path_out, 'valid_part.npy'), valid_part)
    np.save(os.path.join(path_out, 'test_part.npy'), test_part)
else:
    train_part = np.load(os.path.join(path_out, 'train_part.npy'))
    valid_part = np.load(os.path.join(path_out, 'valid_part.npy'))
    test_part = np.load(os.path.join(path_out, 'test_part.npy'))

# check no overlaps
np.intersect1d(train_part,valid_part).shape[0]==0
np.intersect1d(train_part,test_part).shape[0]==0
np.intersect1d(valid_part,test_part).shape[0]==0

x_train = if_x[train_part,:]
y_train = if_y[train_part,:]
x_valid = if_x[valid_part,:]
y_valid = if_y[valid_part,:]
x_test = if_x[test_part,:]
y_test = if_y[test_part,:]

if debug_mode==1:
    x_train = x_train[:FLAGS.batch_size*2,...]
    y_train = y_train[:FLAGS.batch_size*2,...]
    x_valid = x_valid[:FLAGS.batch_size*2,...]
    y_valid = y_valid[:FLAGS.batch_size*2,...]

spe_width = 304

# on affiche le tout pour vérifier qu'il n'y a pas d'erreur:
print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_train.shape))
print('validation set X shape: '+str(x_valid.shape))
print('validation set Y shape: '+str(y_valid.shape))
print('supervision set X shape: '+str(x_test.shape))
print('supervision set Y shape: '+str(y_test.shape))

# %%

# create generator for augmentation
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=False, add_fake_mspikes=None, add_fake_flcs=None, add_noise=None, add_shifting=None, seed=42, output='parallel'):
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
        self.on_epoch_end()

    def __len__(self): # should return the number of batches per epoch
        return self.x.shape[0]//self.batch_size

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
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=FLAGS.batch_size,
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
                        seed = 0,
                        output = 'parallel') # for my own special u-net only, you should set this to false

# if debug_mode==1:
#     # a function for making plots for debugging our augmentation function
#     def plotDebug(xnoaug, ynoaug, xwithaug, ywithaug):
#         from matplotlib import pyplot as plt
#         plt.figure(figsize=(10,9))
#         for i,x,y in [[0,xnoaug,ynoaug],[1,xwithaug,ywithaug]]:
#             for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
#                 if (i==1) & (num==0):
#                     continue
#                 plt.subplot(6,2,i+1+num*2)
#                 plt.plot(np.arange(0,spe_width), x[:,num], '-', color = col, label = lab)
#                 if num==0:
#                     class_map=y.max(axis=-1)
#                 else:
#                     class_map=y[...,num-1]
#                 for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
#                     plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=x[peak_start:peak_end,num], color="red")
#         plt.tight_layout()
    
#     # plot debug
#     batch_ind = 1 # which batch to compute
    
#     # run aug + get unaugmented curves
#     batch_x, batch_y = train_gen.__getitem__(batch_ind)
#     batch_x = batch_x[...,0,0]
#     batch_y = batch_y[...,0,0]
#     noaug_batch_x = train_gen.x[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size]]
#     noaug_batch_y = train_gen.y[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size]]
#     # reshape
#     # [[b+i*FLAGS.batch_size for i in range(5)] for b in range(FLAGS.batch_size)]
#     batch_x = np.transpose(np.stack([batch_x[[b+i*FLAGS.batch_size for i in range(5)],...] for b in range(FLAGS.batch_size)], axis=-1), axes=(2,1,0))
#     batch_y = np.transpose(np.stack([batch_y[[b+i*FLAGS.batch_size for i in range(5)],...] for b in range(FLAGS.batch_size)], axis=-1), axes=(2,1,0))
#     # add fake ref curve for generated x
#     batch_x = np.concatenate([np.expand_dims(batch_x.max(axis=-1), -1), batch_x], -1)
#     # plot
#     # for sample_ind in range(1): # which sample to plot
#     for sample_ind in range(6,train_gen.batch_size): # which sample to plot
#         plotDebug(noaug_batch_x[sample_ind], noaug_batch_y[sample_ind], batch_x[sample_ind], batch_y[sample_ind])


# %%

def curve_iou(y_true, y_pred, smooth = 1e-5):
    trh = tf.cast(tf.greater(y_true, .5), 'double')
    prd = tf.cast(tf.greater(y_pred, .5), 'double')
    i = tf.cast(tf.greater(trh+prd, 1), 'double')
    u = tf.cast(tf.greater(trh+prd, 0), 'double')
    i = tf.reduce_sum(i)
    u = tf.reduce_sum(u)
    return (smooth+i) / (smooth+u)

tf.keras.metrics.curve_iou=curve_iou

# %%

if FLAGS.step=="train":
    
    # enfin, on va pouvoir réaliser le modèle de prédiction
    # réalisé depuis : https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
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
    
    def get_unet_custom(input_signal, blocks=4, n_filters=16, kernel_size=3, dropout=0.5, batchnorm=True, n_classes=5, contract=1, return_model=True, return_sigmoid=True):
        # contracting path
        conv_blocks = []
        
        # encoder
        last_pb = input_signal
        for b in range(blocks+1):
            dropout_mult=1.
            if b==0:
                dropout_mult=.5
            cb = conv1d_block(last_pb, n_filters=n_filters*np.power(2,b), kernel_size=kernel_size, batchnorm=batchnorm)
            conv_blocks.append(cb)
            if b<blocks:
                pb = tf.keras.layers.MaxPooling2D((2,1)) (cb)
                pb = tf.keras.layers.Dropout(dropout*dropout_mult)(pb)
                last_pb = pb
        
        # decoder
        for b in range(blocks):
            ub = tf.keras.layers.Conv2DTranspose(n_filters*np.power(2,blocks-b-1), (kernel_size,1), strides=(2,1), padding='same') (conv_blocks[-1])
            ub = tf.keras.layers.Concatenate() ([ub, conv_blocks[blocks-b-1]])
            ub = tf.keras.layers.Dropout(dropout)(ub)
            cb = conv1d_block(ub, n_filters=n_filters*np.power(2,blocks-b-1), kernel_size=kernel_size, batchnorm=batchnorm)
            conv_blocks.append(cb)
        
        outputs = tf.keras.layers.Conv2D(n_classes, (1,contract), activation='sigmoid') (conv_blocks[-1])
        if return_model:
            model = tf.keras.models.Model(inputs=[input_signal], outputs=[outputs])
            return model
        if return_sigmoid:
            return outputs
        return conv_blocks[-1]
    
    def getUnetWithHP(hp):
        input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_if')
        model = get_unet_custom(input_signal,
                                blocks = hp.Int('blocks', 2, 4, default=4),
                                n_filters = hp.Int('nfilters', min_value=8, max_value=64, step=16, default=8),
                                kernel_size = hp.Int('kernelsize', 3, 7, default=3),
                                dropout = hp.Float('dropout', 0, 0.5, step=0.1, default=0.05),
                                batchnorm = hp.Choice('batchnorm', [1,0], default=1)==1,
                                n_classes = 1,
                                contract = 1,
                                return_model = True,
                                return_sigmoid = True)
        model.compile(loss=FLAGS.loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
        return model
    
    # on crée un callback pour surveiller l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=2),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=min_lr, verbose=2),
        #tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    
    tuner = kt.BayesianOptimization(getUnetWithHP,
                                    objective = 'val_loss',
                                    max_trials=30,
                                    num_initial_points=5,
                                    seed=42,
                                    tune_new_entries=True,
                                    allow_new_entries=True)
    
    # generate validation data
    valid_gen = ITGenerator(x=x_valid, y=y_valid, batch_size=x_valid.shape[0], shuffle=False, output = 'parallel')
    x_valid_tmp, y_valid_tmp = valid_gen.__getitem__(0)
    
    tuner.search(train_gen,
                 epochs=1000,
                 steps_per_epoch=train_gen.__len__(),
                 validation_data=(x_valid_tmp, x_valid_tmp),
                 verbose=2,
                 callbacks = callbacks)
    
    # save hyperparameters results
    best_hyperparams = tuner.get_best_hyperparameters()[0]
    hp_df = pd.DataFrame.from_dict(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values, orient='index')
    hp_df.to_csv(os.path.join(path_out,model_name[:-3]+'.csv'), header=False, sep='\t')

    # reload best model and save it
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(path_out, model_name))
    
    
# %%

# Validation
if FLAGS.step=="test":
    
    # predict
    export_metrics = dict()
    
    model = tf.keras.models.load_model(filepath = os.path.join(path_out,model_name), compile=False)
    
    use_set = 'test'
    if use_set=='train':
        x = x_train
        y = y_train
    elif use_set=='valid':
        x = x_valid
        y = y_valid
    elif use_set=='test':
        x = x_test
        y = y_test
    
    size = x.shape[0]
    start=time.time()
    y_raw_ = model.predict(ITGenerator(x=x, y=y, batch_size=x.shape[0], shuffle=False, output = 'parallel'))
    y_ = y_raw_[...,0,0]
    y_ = np.transpose(np.stack([y_[[b+i*x.shape[0] for i in range(5)],...] for b in range(x.shape[0])], axis=-1), axes=(2,1,0))
    y_ = np.expand_dims(y_, 2)
    # invert predictions !
    y_ref_ = np.max(y_, axis=-1)
    for dim in range(5):
        y_[...,dim] = (1-y_[...,dim])*y_ref_
    end=time.time()
    print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 741 us per sample
    print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.1s per 100 samples
    
    # On va alors déterminer les métriques:
    threshold = .1
    points=np.arange(1,spe_width+1,1)
    pr=np.zeros((y_.shape[0],5))
    iou=np.zeros((y_.shape[0],5))
    iou_tf=np.zeros((y_.shape[0],5))
    for ix in trange(y_.shape[0]):
        for dim in range(5):
            gt = y[ix,:,dim]
            pd_ = (y_[ix,:,0,dim]>threshold)*1
            u = np.sum(gt+pd_>0)
            i = np.sum(gt+pd_==2)
            if np.isfinite(u):
                iou[ix,dim] = i/u
            else:
                iou[ix,dim] = np.nan
            pr[ix,dim] = np.sum(gt==pd_)/spe_width
            # iou_tf[ix,dim] = float(curve_iou(gt.astype('double'),pd_.astype('double')))
            iou_tf[ix,dim] = curve_iou(y[ix,:,dim].astype('double'),y_[ix,:,0,dim].astype('double'))
    
    for k in range(iou.shape[1]):
        print("Mean IoU for fraction '{}': {:.2f} +- {:.2f}".format(['G','A','M','k','l'][k],np.nanmean(iou[:,k]),np.nanstd(iou[:,k])))
        export_metrics['IoU-{}'.format(['G','A','M','k','l'][k])] = np.nanmean(iou[:,k])
    export_metrics['IoU-global'] = np.nanmean(iou)
    
    for k in range(iou_tf.shape[1]):
        print("Mean IoU (tf) for fraction '{}': {:.2f} +- {:.2f}".format(['G','A','M','k','l'][k],np.nanmean(iou_tf[:,k]),np.nanstd(iou_tf[:,k])))
        export_metrics['tfIoU-{}'.format(['G','A','M','k','l'][k])] = np.nanmean(iou_tf[:,k])
    export_metrics['tfIoU-global'] = np.nanmean(iou_tf)
    
    for k in range(pr.shape[1]):
        print("Mean accuracy for fraction '{}': {:.2f} +- {:.2f}".format(['G','A','M','k','l'][k],np.nanmean(pr[:,k]),np.nanstd(pr[:,k])))
    
    if debug_mode==1:
        # Make a function for plotting
        from matplotlib import pyplot as plt
        def plotITPredictions(ix):
            plt.figure(figsize=(14,10))
            plt.subplot(3,1,1)
            # on récupère la class map (binarisée)
            # class_map = y[ix].max(axis=1)
            curve_values = x[ix,:]
            for num,col,lab in zip(range(6), ['black','purple','pink','green','red','blue'], ['Ref','G','A','M','k','l']):
                plt.plot(np.arange(0,spe_width), curve_values[:,num], '-', color = col, label = lab)
            if use_set=='train':
                plt.title('Train set {} (global {})'.format(ix,train_part[ix]))
            elif use_set=='valid':
                plt.title('Valid set {} (global {})'.format(ix,valid_part[ix]))
            elif use_set=='test':
                plt.title('Test set {} (global {})'.format(ix,test_part[ix]))
            plt.legend()
            # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
            #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')
        
            # on plot aussi les autres courbes
            plt.subplot(3,1,2)
            for num,col,lab in zip(range(5), ['purple','pink','green','red','blue'], ['G','A','M','k','l']):
                plt.plot(np.arange(0,spe_width)+1, y[ix,:,num]/5+(4-num)/5, '-', color = col, label = lab)
            plt.ylim(-.05,1.05)
            plt.legend()
            plt.title('Ground truth maps')
            
            plt.subplot(3,1,3)
            for num,col,lab in zip(range(5), ['purple','pink','green','red','blue'], ['G','A','M','k','l']):
                plt.plot(np.arange(0,spe_width)+1, y_[ix,:,0,num]/5+(4-num)/5, '-', color = col, label = lab)
            plt.ylim(-.05,1.05)
            plt.legend()
            plt.title('Predicted maps')
        
        for ix in range(1,3):
            plotITPredictions(ix)
        # Calculons pour chaque pic réel/prédit la concordance
        # TODO : pour l'instant on regarde juste : quand un pic réel, qu'est-ce que c'est comme pic ?
        # A faire : regarder les pics uniquement prédits mais pas réels
    
    threshold=.10
    curve_ids = []
    groundtruth_spikes = []
    predicted_spikes = []
    for ix in trange(x.shape[0]):
        flat_gt = np.zeros_like(y[ix,:,0])
        for i in range(y.shape[-1]):
            flat_gt += y[ix,:,i]*(1+np.power(2,i))
        gt_starts = []
        gt_ends = []
        prev_v = 0
        for i in range(spe_width):
            if flat_gt[i] != prev_v: # changed
                # multiple cases:
                # 0 -> non-zero = enter peak
                if prev_v == 0:
                    gt_starts.append(i)
                # non-zero -> 0 = out of peak
                elif flat_gt[i] == 0:
                    gt_ends.append(i)
                # non-zero -> different non-zero = enter other peak
                else:
                    gt_ends.append(i)
                    gt_starts.append(i)
                prev_v = flat_gt[i]
                
        if len(gt_starts) != len(gt_ends):
            raise Exception('Inconsistent start/end points')
        
        if len(gt_starts)>0:
            # pour chaque pic, on détecte ce que le modèle a rendu a cet endroit comme type d'Ig
            for pstart,pend in zip(gt_starts,gt_ends):
                gt_ig_denom = ''
                if np.sum(y[ix,pstart:pend,:3])>0:
                    HC_gt = int(np.median(np.argmax(y[ix,pstart:pend,:3], axis=1)))
                    gt_ig_denom = ['G','A','M'][HC_gt]
                lC_gt = int(np.median(np.argmax(y[ix,pstart:pend,3:], axis=1)))
                gt_ig_denom += ['k','l'][lC_gt]
                
                pred_ig_denom = ''
                if np.sum(y_[ix,pstart:pend,0,:]>threshold)>0: # un pic a été détecté
                    if np.sum(y_[ix,pstart:pend,0,:3]>threshold)>0:
                        HC_pred = int(np.median(np.argmax(y_[ix,pstart:pend,0,:3], axis=1)))
                        pred_ig_denom = ['G','A','M'][HC_pred]
                    lC_pred = int(np.median(np.argmax(y_[ix,pstart:pend,0,3:], axis=1)))
                    pred_ig_denom += ['k','l'][lC_pred]
                else:
                    pred_ig_denom = 'none'
                    
                groundtruth_spikes.append(gt_ig_denom)
                predicted_spikes.append(pred_ig_denom)
                curve_ids.append(ix)
        else:
            # TODO
            pass
    
    conc_df = pd.DataFrame(dict(ix=curve_ids,
                                true=groundtruth_spikes,
                                pred=predicted_spikes))
        
    print(pd.crosstab(conc_df.true, conc_df.pred))
    
    if debug_mode==1:
        # plot mistakes
        len(conc_df.ix.loc[conc_df.true != conc_df.pred].tolist())

        for i in range(95,105):
            plotITPredictions(conc_df.ix.loc[conc_df.pred == conc_df.true].tolist()[i])
        
        len(conc_df.ix.loc[conc_df.pred == 'none'].tolist())

        plotITPredictions(conc_df.ix.loc[conc_df.pred == 'none'].tolist()[0])
        plotITPredictions(conc_df.ix.loc[conc_df.pred == 'none'].tolist()[1])
        plotITPredictions(conc_df.ix.loc[conc_df.pred == 'none'].tolist()[2])
        plotITPredictions(conc_df.ix.loc[conc_df.pred == 'none'].tolist()[3])
        plotITPredictions(conc_df.ix.loc[conc_df.pred == 'none'].tolist()[4])
        
    print('Global precision: '+str(round(100*np.sum(conc_df.true==conc_df.pred)/conc_df.shape[0],1)))
    for typ in np.unique(conc_df.true):
        subset=conc_df.true==typ
        print('  Precision for type '+typ+': '+str(round(100*np.sum(conc_df.true.loc[subset]==conc_df.pred.loc[subset])/np.sum(subset), 1)))
        export_metrics['Acc-{}'.format(typ)] = 100*np.sum(conc_df.true.loc[subset]==conc_df.pred.loc[subset])/np.sum(subset)
    export_metrics['Acc-global'] = 100*np.sum(conc_df.true==conc_df.pred)/conc_df.shape[0]
    
    export_metrics['Mistakes-total'] = conc_df.loc[conc_df.true!=conc_df.pred,:].shape[0]
    mistakes = conc_df.loc[conc_df.true!=conc_df.pred,'ix'].unique().tolist()
    export_metrics['Mistakes-curves'] = len(mistakes)
    
    # Quand il n'y a pas de pic, qu'est-ce que le modèle renvoie ??
    # -> faire une AUC
    if debug_mode==1:
        from sklearn import metrics
        
        # Uniquement par courbe (max) ou par point
        by_curve = False
        
        if by_curve:
            fpr, tpr, thresholds = metrics.roc_curve(y.max(axis=-1).max(axis=-1), y_[:,:,0,:].max(axis=-1).max(axis=-1))
        else:
            fpr, tpr, thresholds = metrics.roc_curve(y.reshape(-1), y_.reshape(-1))
        
        roc_auc = metrics.auc(fpr, tpr)
        youden = (1-fpr)+tpr-1
        best_threshold = np.where(youden == np.max(youden))[0][0]
        default_threshold = np.argmin(np.abs(thresholds-.5))
        severe_threshold = np.argmin(np.abs(thresholds-.9))
        
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[severe_threshold], tpr[severe_threshold], 'o', color = 'orange')
        plt.plot(fpr[best_threshold], tpr[best_threshold], 'o', color = 'red')
        plt.plot(fpr[default_threshold], tpr[default_threshold], 'o', color = 'blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        if by_curve:
            plt.title('ROC - Monoclonal score for each curve')
        else:
            plt.title('ROC - Monoclonal score for each point')
        plt.legend(loc="lower right")
        tmp_sub=0.01
        plt.text(fpr[severe_threshold]+.01, tpr[severe_threshold]-tmp_sub, 'Severe: ' + str(round(thresholds[severe_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[severe_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[severe_threshold]), 1))) + ')')
        plt.text(fpr[best_threshold]+.01, tpr[best_threshold]+0, 'Optimal: ' + str(round(thresholds[best_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[best_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[best_threshold]), 1))) + ')')
        plt.text(fpr[default_threshold]+.01, tpr[default_threshold]-0, 'Default: ' + str(round(thresholds[default_threshold], 5)) +
                 ' (Se: ' + str(str(round(100*tpr[default_threshold], 1))) + ', Sp: ' + str(str(round(100*(1-fpr[default_threshold]), 1))) + ')')
        plt.show()
    
        threshold = .5
        pd.crosstab(y.max(axis=-1).max(axis=-1)>threshold, y_[:,:,0,:].max(axis=-1).max(axis=-1)>threshold)
        threshold = .9
        pd.crosstab(y.max(axis=-1).max(axis=-1)>threshold, y_[:,:,0,:].max(axis=-1).max(axis=-1)>threshold)
            
    # Export results :
    # Model name, parameters, IoU per plane, global IoU, accuracy per plane, global accuracy, number of mistakes
    metrics_name = log_name[:-3]+'-metrics.pkl'
    with open(os.path.join(path_out,metrics_name), 'wb') as file_pi:
        pickle.dump(export_metrics, file_pi)
    
    
    # plotITPredictions(mistakes[0])
    # plotITPredictions(mistakes[1])
    # plotITPredictions(mistakes[2])
    # plotITPredictions(mistakes[3])
    # plotITPredictions(mistakes[4])
    # plotITPredictions(mistakes[5])
    # plotITPredictions(mistakes[6])
    # plotITPredictions(mistakes[7])
    # plotITPredictions(mistakes[8])
    
    
    # y_true = y[[0]].reshape((1,160,1,5)).astype('double')
    # y_pred = y_[[0]].astype('double')
    
    # np.mean(jaccard_distance_loss(y_true, y_pred))
    
    # curve_iou(y_true, y_pred)
    
    # y_true_flat = y_true[0,:,0,:]
    # y_pred_flat = y_pred[0,:,0,:]
    
# %%
    
if FLAGS.step=="aggreg":
    
    if debug_mode==1:
        models_logs = [n for n in os.listdir(path_out) if n[-12:]=='-metrics.pkl']
        models_logs_data = []
        for model_logs in models_logs:
            with open(os.path.join(path_out,model_logs), 'rb') as file_pi:
                tmp_dict = pickle.load(file_pi)
                tmp_dict['model'] = model_logs[:-13]
                models_logs_data.append(tmp_dict)
        models_logs_df = pd.DataFrame(models_logs_data)
        models_logs_df.to_excel(os.path.join(path_out,'validationv2.xlsx'))


    



    



    



    



    



    



    



    



    



    



    



