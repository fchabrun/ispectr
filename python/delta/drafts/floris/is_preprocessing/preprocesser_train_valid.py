# -*- coding: utf-8 -*-
"""
Created on 07/08/21

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

import argparse
import time
from tqdm import trange
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
# parser.add_argument("--step", type=str, default='test')
# parser.add_argument("--arch", type=str, default='parallel_unetv1wide')
parser.add_argument("--loss", type=str, default='binary_crossentropy')
# parser.add_argument("--part", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size

model_name = 'is_preprocesser_v1.h5'
log_name = model_name[:-2]+"pkl"

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if debug_mode == 1:
    path_in = 'C:/Users/admin/Documents/Capillarys/data/2021/ifs'
    path_out = 'C:/Users/admin/Documents/Capillarys/temp2021'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/out'
    
# debug_mode=0
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

part_rng = np.random.RandomState(seed=42)

train_part = part_rng.choice(a = np.arange(if_x.shape[0]), size = int(if_x.shape[0]*.9), replace = False)
valid_part = np.setdiff1d(np.arange(if_x.shape[0]), train_part)

x_train = if_x[train_part,:]
y_train = if_y[train_part,:]
x_valid = if_x[valid_part,:]
y_valid = if_y[valid_part,:]

# KEEP ONLY FIRST DIMENSION OF X DATA (i.e. ELP, not G, A, M, k, l)
x_train = x_train[...,0]
x_valid = x_valid[...,0]

# GET "MAX" OF ALL DIMENSIONS FOR Y DATA
y_train = y_train.max(axis=-1)
y_valid = y_valid.max(axis=-1)

spe_width = 304

# on affiche le tout pour vérifier qu'il n'y a pas d'erreur:
print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_train.shape))
print('validation set X shape: '+str(x_valid.shape))
print('validation set Y shape: '+str(y_valid.shape))

# %%

# create generator for augmentation
class ELPGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=True, add_fake_mspikes=None, add_fake_flcs=None, add_noise=None, add_shifting=None, seed=42):
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
                        pos=self.rng.choice(np.arange(self.add_fake_mspikes['minpos'],self.add_fake_mspikes['maxpos']))
                        height=self.rng.uniform(self.add_fake_mspikes['minheight'],self.add_fake_mspikes['maxheight'])
                        width=self.rng.uniform(self.add_fake_mspikes['minwidth'],self.add_fake_mspikes['maxwidth'])
                        curve_with_mspike = norm.pdf(np.arange(304),pos,width)*height
                        batch_y[i,...] = np.maximum(batch_y[i,...], ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1)
                        batch_x[i,...] += curve_with_mspike + curve_with_mspike*np.random.normal(0,.01,curve_with_mspike.shape)
        if self.add_fake_flcs is not None:
            freq=self.add_fake_flcs['freq'] # between 0. and 1., how often do wee add noise
            for i in range(batch_x.shape[0]):
                if self.rng.random(1) < freq:
                    # add a fake mspike
                    for ms in range(self.rng.choice(np.arange(self.add_fake_flcs['mincount'],self.add_fake_flcs['maxcount']+1))):
                        # choose random dims
                        pos=self.rng.choice(np.arange(self.add_fake_flcs['minpos'],self.add_fake_flcs['maxpos']))
                        height=self.rng.uniform(self.add_fake_flcs['minheight'],self.add_fake_flcs['maxheight'])
                        width=self.rng.uniform(self.add_fake_flcs['minwidth'],self.add_fake_flcs['maxwidth'])
                        curve_with_mspike = norm.pdf(np.arange(304),pos,width)*height
                        batch_y[i,...] = np.maximum(batch_y[i,...], ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1)
                        batch_x[i,...] += curve_with_mspike + curve_with_mspike*np.random.normal(0,.01,curve_with_mspike.shape)
        
        return np.expand_dims(np.expand_dims(batch_x, axis=-1), axis=-1), np.expand_dims(np.expand_dims(batch_y, axis=-1), axis=-1)
    
# create our train generator
train_gen = ELPGenerator(x=x_train, y=y_train, batch_size=FLAGS.batch_size,
                        # add_fake_mspikes=None,
                        add_fake_mspikes=dict(freq=.5,
                                              minpos=180, # 180 is nice
                                              maxpos=251, # 251 is nice
                                              minheight=.5, # .5 is nice
                                              maxheight=8., # 8 is nice
                                              minwidth=3.5, # 3.5 is nice
                                              maxwidth=4.5, # 4.5 is nice
                                              mincount=1,
                                              maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                        add_fake_flcs=None,
                        # add_fake_flcs=dict(freq=.1,
                        #                    minpos=100, # 180 is nice
                        #                    maxpos=251, # 251 is nice
                        #                    minheight=.4, # .5 is nice
                        #                    maxheight=.5, # 8 is nice
                        #                    minwidth=2.5, # 3.5 is nice
                        #                    maxwidth=3.5, # 4.5 is nice
                        #                    mincount=1,
                        #                    maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                        # add_noise=None,
                        add_noise=dict(freq=.2, minstd=.001, maxstd=.002),
                        # add_shifting=None,
                        add_shifting=dict(freq=.8, min=1, max=4),
                        seed = 0) # for my own special u-net only, you should set this to false

validaug_gen = ELPGenerator(x=x_valid, y=y_valid, batch_size=FLAGS.batch_size,
                            add_fake_mspikes=None,
                            # add_fake_mspikes=dict(freq=.5,
                            #                       minpos=180, # 180 is nice
                            #                       maxpos=251, # 251 is nice
                            #                       minheight=.2, # .5 is nice
                            #                       maxheight=5, # 8 is nice
                            #                       minwidth=3.5, # 3.5 is nice
                            #                       maxwidth=4.5, # 4.5 is nice
                            #                       mincount=1,
                            #                       maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                            add_fake_flcs=None,
                            # add_fake_flcs=dict(freq=.1,
                            #                    minpos=100, # 180 is nice
                            #                    maxpos=251, # 251 is nice
                            #                    minheight=.4, # .5 is nice
                            #                    maxheight=.5, # 8 is nice
                            #                    minwidth=2.5, # 3.5 is nice
                            #                    maxwidth=3.5, # 4.5 is nice
                            #                    mincount=1,
                            #                    maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                            add_noise=None,
                            # add_noise=dict(freq=.2, minstd=.001, maxstd=.002),
                            add_shifting=None,
                            # add_shifting=dict(freq=.8, min=1, max=4),
                            seed = 0,
                            shuffle=False) # for my own special u-net only, you should set this to false

if False:
    # a function for making plots for debugging our augmentation function
    batch_ind = 0 # which batch to compute
    # run aug + get unaugmented curves
    batch_x, batch_y = train_gen.__getitem__(batch_ind)
    batch_x = batch_x[...,0,0]
    batch_y = batch_y[...,0,0]
    # get raw data for this batch
    noaug_batch_x = train_gen.x[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size],...]
    noaug_batch_y = train_gen.y[train_gen.indexes[batch_ind*train_gen.batch_size:(batch_ind+1)*train_gen.batch_size],...]
    # plot
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12,10))
    for sample_ind in range(batch_x.shape[0]): # which sample to plot
        plt.subplot(batch_x.shape[0],4,sample_ind*4+1)
        plt.plot(np.arange(0,spe_width), noaug_batch_x[sample_ind,...])
        plt.title("Raw (no augmentation)")
        plt.tight_layout()
        plt.subplot(batch_x.shape[0],4,sample_ind*4+2)
        plt.plot(np.arange(0,spe_width), noaug_batch_y[sample_ind,...])
        plt.title("Output (no augmentation)")
        plt.tight_layout()
        plt.subplot(batch_x.shape[0],4,sample_ind*4+3)
        plt.plot(np.arange(0,spe_width), batch_x[sample_ind,...])
        plt.title("Processed (with augmentation)")
        plt.tight_layout()
        plt.subplot(batch_x.shape[0],4,sample_ind*4+4)
        plt.plot(np.arange(0,spe_width), batch_y[sample_ind,...])
        plt.title("Output (with augmentation)")
        plt.tight_layout()
    plt.tight_layout()

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
    
    # on crée un callback pour surveiller l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=min_lr, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    
    input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_elp')
    model = get_unet_custom(input_signal, blocks=4, n_filters=32, kernel_size=3, dropout=0.05, batchnorm=True, n_classes=1)
    model.compile(loss=FLAGS.loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    
    print(model.summary())

    N_EPOCHS = 1000

    print('Setting batch size to: '+str(BATCH_SIZE))
    print('Setting maximal number of epochs to: '+str(N_EPOCHS))
    
    results = model.fit(train_gen,
                        batch_size = BATCH_SIZE,
                        epochs = N_EPOCHS,
                        callbacks = callbacks,
                        verbose = 2-debug_mode,
                        validation_data = validaug_gen,
                        validation_steps = validaug_gen.__len__())
    
    # enfin, on voudra sauvegarder les résultats
    # Save history
    with open(os.path.join(path_out,log_name), 'wb') as file_pi:
        pickle.dump(results.history, file_pi)
    
# %%

# Validation
if FLAGS.step=="test":
    
    # reload desired model
    model = tf.keras.models.load_model(filepath = os.path.join(path_out,model_name), compile=False)
    # model.load_weights(os.path.join(path_out,"working_weights.h5"), by_name=True)
    # model = tf.keras.models.load_model(filepath = r"C:\Users\admin\Documents\Capillarys\temp2021\kerastuner_unetparallel\segifaug-parallelunetv1-loss-binary_crossentropy.h5", compile=False)

    # predict
    export_metrics = dict()
    
    # use_set = 'train'
    use_set = 'valid'
    if use_set=='train':
        x = x_train
        y = y_train
    elif use_set=='valid':
        x = x_valid
        y = y_valid
    
    size = x.shape[0]
    start=time.time()
    
    temp_gen = ELPGenerator(x=x, y=y, batch_size=x.shape[0], shuffle=False)
    y_raw_ = model.predict(temp_gen)
    y_ = y_raw_[...,0,0]
            
    end=time.time()
    print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 1.385 ms per sample
    print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.1s per 100 samples
    
    # On va alors déterminer les métriques:
        
    # 1. AUC
    if debug_mode==1:
        from sklearn import metrics
        
        fpr, tpr, thresholds = metrics.roc_curve(y.flatten(), y_.flatten())
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
        
        # default threshold looks nice
    
    # 2. IOU, Accuracy...
    threshold = .5
    points=np.arange(1,spe_width+1,1)
    pr=np.zeros((y_.shape[0]))
    iou=np.zeros((y_.shape[0]))
    iou_tf=np.zeros((y_.shape[0]))
    for ix in trange(y_.shape[0]):
        gt = y[ix,:]
        pd_ = (y_[ix,:]>threshold)*1
        u = np.sum(gt+pd_>0)
        i = np.sum(gt+pd_==2)
        if np.isfinite(u):
            iou[ix] = i/u
        else:
            iou[ix] = np.nan
        pr[ix] = np.sum(gt==pd_)/spe_width
        # iou_tf[ix,dim] = float(curve_iou(gt.astype('double'),pd_.astype('double')))
        iou_tf[ix] = curve_iou(y[ix,:].astype('double'),y_[ix,:].astype('double'))
    
    print("")
    
    print("Mean IoU: {:.3f} +- {:.2f}".format(np.nanmean(iou),np.nanstd(iou))) # 0.918 +- 0.15
    print("Mean IoU (tf): {:.3f} +- {:.2f}".format(np.nanmean(iou_tf),np.nanstd(iou_tf))) # 0.919 +- 0.15
    print("Mean accuracy: {:.3f} +- {:.2f}".format(np.nanmean(pr),np.nanstd(pr))) # 0.996 +- 0.01
    
    export_metrics['IoU'] = np.nanmean(iou)
    export_metrics['tfIoU'] = np.nanmean(iou_tf)
    export_metrics['Acc'] = np.nanmean(pr)
    
    # 3. PLOTS
    
    if debug_mode==1:
        def plotELPPred(ix):
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12,6))
            #
            plt.subplot(3,1,1)
            plt.plot(np.arange(0,spe_width), x[ix,...])
            plt.title("Input curve")
            #
            plt.subplot(3,1,2)
            plt.plot(np.arange(0,spe_width), y[ix,...])
            plt.title("Ground truth")
            #
            plt.subplot(3,1,3)
            plt.plot(np.arange(0,spe_width), y_[ix,...])
            plt.title("Prediction")
            plt.tight_layout()
        
        plotELPPred(3)
    
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


    



    



    



    



    



    



    



    



    



    



    



