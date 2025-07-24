# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# Ce modèle va segmenter la courbe
# On va prédire pour chaque pixel de la courbe (300 pixels) si ce pixel est :
# zone A
# zone a1
# zone a2
# zone b1
# zone b2
# zone gamma
# pic

# la question se pose de savoir si on considère un pic comme fraction+pic (ie: gamma ET pic)
# ou uniquement comme le pic, et pas une fraction

# on va commencer par charger les données

print('Starting IF classification script...')

import argparse
import time
from tqdm import trange
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# TODO -> il y a un bug (qui marche, mais un bug néanmoins): on instancie 5 (6) u-nets différents, au lieu d'un seul avec les mêmes poids

# TODO prendre modèle pré-entraîné sur pics ? (taille doit être 304 par contre)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--step", type=str, default='init')
parser.add_argument("--arch", type=str, default='unetD5Pchannel')
parser.add_argument("--loss", type=str, default='binary_crossentropy')
parser.add_argument("--lstm_order", type=str, default='ref_first')
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--norm", type=int, default=0)
parser.add_argument("--cut", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--weights", type=str, default='')
parser.add_argument("--weights", type=str, default='segmentation_spikes_best_full_model_2020-batchsize-32.h5')

# parser.add_argument("--gan_training", type=int, default=0)
FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
norm = FLAGS.norm==1
cut = FLAGS.cut==1
do_part = FLAGS.part==1
weights = FLAGS.weights
if len(weights)==0:
    weights=None

if FLAGS.arch in ('lstm','multi_lstm','lstm_seq'):
    model_name = 'segif-{}-batchsize-{}-norm-{}-cut{}-order-{}-loss-{}.h5'.format(FLAGS.arch,FLAGS.batch_size,FLAGS.norm,FLAGS.cut,FLAGS.lstm_order,FLAGS.loss)
elif FLAGS.arch in ('unet6Pdim', 'unet6Pchannel', 'unetD5Pdim', 'unetD5Pchannel'):
    if weights is not None:
        base_lr = 1e-5 # reduce lr if pre-trained
        model_name = 'segif-{}-batchsize-{}-norm-{}-cut{}-loss-{}-weights-transfer.h5'.format(FLAGS.arch,FLAGS.batch_size,FLAGS.norm,FLAGS.cut,FLAGS.loss)
    else:
        model_name = 'segif-{}-batchsize-{}-norm-{}-cut{}-loss-{}-weights-init.h5'.format(FLAGS.arch,FLAGS.batch_size,FLAGS.norm,FLAGS.cut,FLAGS.loss)
else:
    model_name = 'segif-{}-batchsize-{}-norm-{}-cut{}-loss-{}.h5'.format(FLAGS.arch,FLAGS.batch_size,FLAGS.norm,FLAGS.cut,FLAGS.loss)

log_name = model_name[:-2]+"pkl"

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if debug_mode == 1:
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = 'C:/Users/admin/Documents/Capillarys/temp2021'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/out'

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

spe_width = 304

if cut:
    # cut first half
    spe_width=160
    x_train = x_train[:,-160:,:]
    x_valid = x_valid[:,-160:,:]
    x_test = x_test[:,-160:,:]
    y_train = y_train[:,-160:,:]
    y_valid = y_valid[:,-160:,:]
    y_test = y_test[:,-160:,:]

if norm:
    # compute & apply mean & std
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    # replace 0's
    x_std[x_std==0] = 1
    x_train = (x_train-x_mean)/x_std
    x_valid = (x_valid-x_mean)/x_std
    x_test = (x_test-x_mean)/x_std

# on affiche le tout pour vérifier qu'il n'y a pas d'erreur:
print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_train.shape))
print('validation set X shape: '+str(x_valid.shape))
print('validation set Y shape: '+str(y_valid.shape))
print('supervision set X shape: '+str(x_test.shape))
print('supervision set Y shape: '+str(y_test.shape))

def jaccard_distance_loss(y_true, y_pred, smooth=100, ax=-1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=ax)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=ax)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def curve_iou(y_true, y_pred, smooth = 1e-5):
    trh = tf.cast(tf.greater(y_true, .5), 'double')
    prd = tf.cast(tf.greater(y_pred, .5), 'double')
    i = tf.cast(tf.greater(trh+prd, 1), 'double')
    u = tf.cast(tf.greater(trh+prd, 0), 'double')
    i = tf.reduce_sum(i)
    u = tf.reduce_sum(u)
    return (smooth+i) / (smooth+u)

if False:
    y = y_train
    
    curve_iou(tf.cast(y[[0],:,:], 'double'),tf.cast(y[[0],:,:], 'double'))
    
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:], ax=0)) # 0
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:]/2, ax=0)) # 11
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:]/1e5, ax=0)) # 22
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,::-1], ax=0)) # 22
    np.sum(jaccard_distance_loss(y[[0],:,[0]], y[[0],:,[2]], ax=0)) # 11
    np.sum(jaccard_distance_loss(y[[0],:,[2]], y[[0],:,[0]], ax=0)) # 11
    
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:], ax=-1)) # 0
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:]/2, ax=-1)) # 11
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,:]/1e5, ax=-1)) # 22
    np.sum(jaccard_distance_loss(y[[0],:,:],y[[0],:,::-1], ax=-1)) # 22
    np.sum(jaccard_distance_loss(y[[0],:,[0]], y[[0],:,[2]], ax=-1)) # 11
    np.sum(jaccard_distance_loss(y[[0],:,[2]], y[[0],:,[0]], ax=-1)) # 11
    
    np.sum(jaccard_distance_loss(y[[0],:,:], y[[0],:,:])) # should be 0 -> perfect
    np.sum(jaccard_distance_loss(y[[0],:,:], y[[0],:,:]/1e5)) # 25 should be a litle higher : perfect but positive results < 0.5
    np.sum(jaccard_distance_loss(y[[0],:,:], y[[0],:,::-1])) # 25 should be something like 50%

    np.sum(jaccard_distance_loss(y[[0],:,0], y[[0],:,2])) # should be high : saw peak, but no peak
    np.sum(jaccard_distance_loss(y[[0],:,2], y[[0],:,0])) # should be high : did not see peak
    
    np.sum(jaccard_distance_loss(y[[0],:,0], y[[0],:,0])) # should be 0
    np.sum(jaccard_distance_loss(y[[0],:,1], y[[0],:,1])) # should be 0
    np.sum(jaccard_distance_loss(y[[0],:,2], y[[0],:,2])) # should be 0
    np.sum(jaccard_distance_loss(y[[0],:,3], y[[0],:,3])) # should be 0
    np.sum(jaccard_distance_loss(y[[0],:,4], y[[0],:,4])) # should be 0
    
    # generate fake preds
    y_true = y_train[:8,...]
    y_pred = y_true + np.random.normal(loc = 0, scale = .05, size = y_true.shape)
    y_pred[y_pred<0] = 0
    y_pred[y_pred>1] = 1

    np.mean(jaccard_distance_loss(y_true,y_pred)) # 0.1
    curve_iou(y_true, y_pred) # 1.0

tf.keras.losses.jaccard_distance_loss = jaccard_distance_loss
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
    
    # on crée un callback pour surveiller l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=min_lr, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    
    if FLAGS.loss == 'jaccard':
        loss = jaccard_distance_loss
    else:
        loss = FLAGS.loss
    
    if FLAGS.arch=="multi_channel":
        # create model from scratch
        input_signal = tf.keras.layers.Input((spe_width, 1, 6), name='input_if')
        model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=5)
        # model.compile(loss=jaccard_distance_loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou,curve_accuracy])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=="multi_dim":
        input_signal = tf.keras.layers.Input((spe_width, 6, 1), name='input_if')
        model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=5, contract=6)
        # model.compile(loss=jaccard_distance_loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou,curve_accuracy])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='multi_model':
        input_signals = [tf.keras.layers.Input((spe_width, 1, 2), name='input_{}'.format(input_name)) for input_name in ('G','A','M','k','l')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False) for input_signal in input_signals]
        # cnn output ?
        final_layer = tf.keras.layers.Concatenate() (core_models)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[final_layer])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='multi_model_multi_dim':
        input_signals = [tf.keras.layers.Input((spe_width, 2, 1), name='input_{}'.format(input_name)) for input_name in ('G','A','M','k','l')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=2, return_model=False) for input_signal in input_signals]
        # cnn output ?
        final_layer = tf.keras.layers.Concatenate() (core_models)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[final_layer])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='multi_lstm':
        inputs = []
        outputs = []
        for input_name in ('G','A','M','k','l'):
            input_sequence = tf.keras.layers.Input((2, spe_width, 1, 1), name='input_sequence_{}'.format(input_name))
            input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_signal_{}'.format(input_name))
            core_model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=True)
            x = tf.keras.layers.TimeDistributed(core_model) (input_sequence)
            x = tf.keras.layers.Reshape((2,spe_width)) (x)
            x = tf.keras.layers.LSTM(spe_width) (x)
            x = tf.keras.layers.Reshape((spe_width,1,1)) (x)
            inputs.append(input_sequence)
            outputs.append(x)
        # cnn output ?
        final_layer = tf.keras.layers.Concatenate() (outputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=[final_layer])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='lstm':
        input_sequence = tf.keras.layers.Input((6, spe_width, 1, 1), name='input_sequence')
        input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_signal')
        core_model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=True)
        x = tf.keras.layers.TimeDistributed(core_model) (input_sequence)
        x = tf.keras.layers.Reshape((6,spe_width)) (x)
        x = tf.keras.layers.LSTM(5*spe_width) (x)
        x = tf.keras.layers.Reshape((spe_width,1,5)) (x)
        model = tf.keras.models.Model(inputs=[input_sequence], outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='lstm_seq':
        input_sequence = tf.keras.layers.Input((6, spe_width, 1, 1), name='input_sequence')
        input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_signal')
        core_model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=True)
        x = tf.keras.layers.TimeDistributed(core_model) (input_sequence)
        x = tf.keras.layers.Reshape((6,spe_width)) (x)
        x = tf.keras.layers.LSTM(spe_width, input_shape = (6, spe_width, 1, 1), return_sequences=True) (x)
        model = tf.keras.models.Model(inputs=[input_sequence], outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unet6channel':
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('Ref','G','A','M','k','l')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-1) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unet6dim':
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('Ref','G','A','M','k','l')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-2) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,6), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unetD5channel':
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('dG','dA','dM','dk','dl')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-1) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unetD5dim':
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('dG','dA','dM','dk','dl')]
        core_models = [get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-2) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,5), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unet6Pchannel':
        if weights is not None:
            assert os.path.exists(os.path.join(path_in,weights)), 'Pre-trained model weights not found at path: "{}"'.format(os.path.join(path_in,weights))
            core_model = tf.keras.models.load_model(os.path.join(path_in,weights), compile=False)
            core_model = tf.keras.models.Model(inputs=core_model.inputs, outputs=[core_model.layers[-2].output])
        else:
            sub_input = tf.keras.layers.Input((spe_width, 1, 1))
            core_model = tf.keras.models.Model(inputs=[sub_input], outputs=[get_unet(sub_input, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False)])
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('Ref','G','A','M','k','l')]
        core_models = [core_model (input_signal) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-1) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unet6Pdim':
        if weights is not None:
            assert os.path.exists(os.path.join(path_in,weights)), 'Pre-trained model weights not found at path: "{}"'.format(os.path.join(path_in,weights))
            core_model = tf.keras.models.load_model(os.path.join(path_in,weights), compile=False)
            core_model = tf.keras.models.Model(inputs=core_model.inputs, outputs=[core_model.layers[-2].output])
        else:
            sub_input = tf.keras.layers.Input((spe_width, 1, 1))
            core_model = tf.keras.models.Model(inputs=[sub_input], outputs=[get_unet(sub_input, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False)])
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('Ref','G','A','M','k','l')]
        core_models = [core_model (input_signal) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-2) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,6), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unetD5Pchannel': # same, but parallel
        if weights is not None:
            assert os.path.exists(os.path.join(path_in,weights)), 'Pre-trained model weights not found at path: "{}"'.format(os.path.join(path_in,weights))
            core_model = tf.keras.models.load_model(os.path.join(path_in,weights), compile=False)
            core_model = tf.keras.models.Model(inputs=core_model.inputs, outputs=[core_model.layers[-2].output])
        else:
            sub_input = tf.keras.layers.Input((spe_width, 1, 1))
            core_model = tf.keras.models.Model(inputs=[sub_input], outputs=[get_unet(sub_input, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False)])
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('dG','dA','dM','dk','dl')]
        core_models = [core_model (input_signal) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-1) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    elif FLAGS.arch=='unetD5Pdim':
        if weights is not None:
            assert os.path.exists(os.path.join(path_in,weights)), 'Pre-trained model weights not found at path: "{}"'.format(os.path.join(path_in,weights))
            core_model = tf.keras.models.load_model(os.path.join(path_in,weights), compile=False)
            core_model = tf.keras.models.Model(inputs=core_model.inputs, outputs=[core_model.layers[-2].output])
        else:
            sub_input = tf.keras.layers.Input((spe_width, 1, 1))
            core_model = tf.keras.models.Model(inputs=[sub_input], outputs=[get_unet(sub_input, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1, contract=1, return_model=False, return_sigmoid=False)])
        input_signals = [tf.keras.layers.Input((spe_width, 1, 1), name='input_{}'.format(input_name)) for input_name in ('dG','dA','dM','dk','dl')]
        core_models = [core_model (input_signal) for input_signal in input_signals]
        # either concat on last axis or concat on previous axis
        x = tf.keras.layers.Concatenate(axis=-2) (core_models)
        x = tf.keras.layers.Conv2D(5, (1,5), activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=input_signals, outputs=[x])
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    else:
        raise Exception('Not coded yet')
    
    print(model.summary())
    
    # Finally, save means and standard deviations
    # dataset_parameters = {'x_mean': x_mean.tolist(), 'x_sd':x_sd.tolist()}
    # json.dump(dataset_parameters, codecs.open(path_out+'segmentation_cnn_parameters.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) 
    
    # on retire les variables inutilisées
    # del f, f_train, f_valid, ix, raw, s, s_train, s_valid, spike_range, train_part, valid_part
    
    # on peut alors entraîner le modèle
    N_EPOCHS = 1000
    # pour le debugging, on peut retirer des échantillons pour accélérer le test
    # if debug_mode==1:
    #     BATCH_SIZE=8
    #     N_EPOCHS=10
    #     sz=BATCH_SIZE*10
    #     x_train=x_train[:sz,...]
    #     x_super=x_super[:sz,...]
    #     y_map_train=y_map_train[:sz,...]
    #     y_map_super=y_map_super[:sz,...]
    print('Setting batch size to: '+str(BATCH_SIZE))
    print('Setting maximal number of epochs to: '+str(N_EPOCHS))

# %%

# A faire pour améliorer le modèle
# Normaliser moyenne/sd au lieu de 0-1 ? (car alb +++)
# Normaliser 0-1 mais cut 150 premiers points ?
# Changer l'architecture
# Changer les loss/métriques : il y a l'air d'avoir un soucis ?

if FLAGS.step == "train":

    if FLAGS.arch=='multi_channel':
        results = model.fit(x_train.reshape( (x_train.shape[0], spe_width, 1, 6) ),
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid.reshape( (x_valid.shape[0], spe_width, 1, 6) ),
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
    elif FLAGS.arch=='multi_dim':
        results = model.fit(x_train.reshape( (x_train.shape[0], spe_width, 6, 1) ),
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid.reshape( (x_valid.shape[0], spe_width, 6, 1) ),
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
    elif FLAGS.arch=='multi_model':
        results = model.fit([x_train[:,:,[0,input_channel]].reshape( (x_train.shape[0], spe_width, 1, 2) ) for input_channel in range(1,6)],
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=([x_valid[:,:,[0,input_channel]].reshape( (x_valid.shape[0], spe_width, 1, 2) ) for input_channel in range(1,6)],
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
    elif FLAGS.arch=='multi_model_multi_dim':
        results = model.fit([x_train[:,:,[0,input_channel]].reshape( (x_train.shape[0], spe_width, 2, 1) ) for input_channel in range(1,6)],
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=([x_valid[:,:,[0,input_channel]].reshape( (x_valid.shape[0], spe_width, 2, 1) ) for input_channel in range(1,6)],
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
    elif FLAGS.arch=='multi_lstm':
        if FLAGS.lstm_order == 'ref_first':
            x_train_for_multi_lstm = [np.concatenate([x_train[:,:,0].reshape((x_train.shape[0], 1, spe_width, 1, 1)), x_train[:,:,input_channel].reshape((x_train.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
            x_valid_for_multi_lstm = [np.concatenate([x_valid[:,:,0].reshape((x_valid.shape[0], 1, spe_width, 1, 1)), x_valid[:,:,input_channel].reshape((x_valid.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
        elif FLAGS.lstm_order == 'ref_last':
            x_train_for_multi_lstm = [np.concatenate([x_train[:,:,input_channel].reshape((x_train.shape[0], 1, spe_width, 1, 1)), x_train[:,:,0].reshape((x_train.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
            x_valid_for_multi_lstm = [np.concatenate([x_valid[:,:,input_channel].reshape((x_valid.shape[0], 1, spe_width, 1, 1)), x_valid[:,:,0].reshape((x_valid.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        
        results = model.fit(x_train_for_multi_lstm,
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid_for_multi_lstm,
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
        
    elif FLAGS.arch=='lstm':
        x_train_for_lstm = []
        x_valid_for_lstm = []
        if FLAGS.lstm_order == 'ref_first':
            for i in range(6):
                x_train_for_lstm.append(x_train[:,:,i].reshape( (x_train.shape[0], 1, spe_width, 1, 1)))
                x_valid_for_lstm.append(x_valid[:,:,i].reshape( (x_valid.shape[0], 1, spe_width, 1, 1)))
        elif FLAGS.lstm_order == 'ref_last':
            for i in (1,2,3,4,5,0):
                x_train_for_lstm.append(x_train[:,:,i].reshape( (x_train.shape[0], 1, spe_width, 1, 1)))
                x_valid_for_lstm.append(x_valid[:,:,i].reshape( (x_valid.shape[0], 1, spe_width, 1, 1)))
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        x_train_for_lstm = np.concatenate(x_train_for_lstm, axis=1)
        x_valid_for_lstm = np.concatenate(x_valid_for_lstm, axis=1)
     
        results = model.fit(x_train_for_lstm,
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid_for_lstm,
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
        
    elif FLAGS.arch=='lstm_seq':
        x_train_for_lstm = []
        x_valid_for_lstm = []
        if FLAGS.lstm_order == 'ref_first':
            for i in range(6):
                x_train_for_lstm.append(x_train[:,:,i].reshape( (x_train.shape[0], 1, spe_width, 1, 1)))
                x_valid_for_lstm.append(x_valid[:,:,i].reshape( (x_valid.shape[0], 1, spe_width, 1, 1)))
        elif FLAGS.lstm_order == 'ref_last':
            for i in (1,2,3,4,5,0):
                x_train_for_lstm.append(x_train[:,:,i].reshape( (x_train.shape[0], 1, spe_width, 1, 1)))
                x_valid_for_lstm.append(x_valid[:,:,i].reshape( (x_valid.shape[0], 1, spe_width, 1, 1)))
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        x_train_for_lstm = np.concatenate(x_train_for_lstm, axis=1)
        x_valid_for_lstm = np.concatenate(x_valid_for_lstm, axis=1)
        
        y_train_for_lstm = np.stack([y_train[:,:,input_channel] for input_channel in range(5)], axis=1)
        y_valid_for_lstm = np.stack([y_valid[:,:,input_channel] for input_channel in range(5)], axis=1)
        if FLAGS.lstm_order == 'ref_first':
            y_train_for_lstm = np.concatenate([y_train_for_lstm.max(axis=1).reshape((y_train.shape[0], 1, spe_width)), y_train_for_lstm], axis=1)
            y_valid_for_lstm = np.concatenate([y_valid_for_lstm.max(axis=1).reshape((y_valid.shape[0], 1, spe_width)), y_valid_for_lstm], axis=1)
        elif FLAGS.lstm_order == 'ref_last':
            y_train_for_lstm = np.concatenate([y_train_for_lstm, y_train_for_lstm.max(axis=1).reshape((y_train.shape[0], 1, spe_width))], axis=1)
            y_valid_for_lstm = np.concatenate([y_valid_for_lstm, y_valid_for_lstm.max(axis=1).reshape((y_valid.shape[0], 1, spe_width))], axis=1)
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
     
        results = model.fit(x_train_for_lstm,
                            y_train_for_lstm,
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid_for_lstm,
                                             y_valid_for_lstm))
        
    elif FLAGS.arch in ('unet6channel', 'unet6dim', 'unet6Pchannel', 'unet6Pdim', ):
        results = model.fit([x_train[:,:,input_channel].reshape( (x_train.shape[0], spe_width, 1, 1) ) for input_channel in range(6)],
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=([x_valid[:,:,input_channel].reshape( (x_valid.shape[0], spe_width, 1, 1) ) for input_channel in range(6)],
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
        
    elif FLAGS.arch in ('unetD5channel', 'unetD5dim', 'unetD5Pchannel', 'unetD5Pdim', ):
        # delta entre ref et g/a/m/k/l
        x_train_tmp = [(x_train[:,:,0]-x_train[:,:,input_channel]).reshape( (x_train.shape[0], spe_width, 1, 1) ) for input_channel in range(1,6)]
        x_valid_tmp = [(x_valid[:,:,0]-x_valid[:,:,input_channel]).reshape( (x_valid.shape[0], spe_width, 1, 1) ) for input_channel in range(1,6)]
        results = model.fit(x_train_tmp,
                            y_train.reshape( (y_train.shape[0], spe_width, 1, 5) ),
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            callbacks=callbacks,
                            verbose=2,
                            validation_data=(x_valid_tmp,
                                             y_valid.reshape( (y_valid.shape[0], spe_width, 1, 5) )))
    
    # enfin, on voudra sauvegarder les résultats
    # Save history
    with open(os.path.join(path_out,log_name), 'wb') as file_pi:
        pickle.dump(results.history, file_pi)
    

# plt.plot(results.history['loss'])
# plt.plot(results.history['val_loss'])
    
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
    if FLAGS.arch == 'multi_channel':
        y_=model.predict(x.reshape( (x.shape[0], spe_width, 1, 6) ))
    elif FLAGS.arch == 'multi_dim':
        y_=model.predict(x.reshape( (x.shape[0], spe_width, 6, 1) ))
    elif FLAGS.arch == 'multi_model':
        y_=model.predict([x[:,:,[0,input_channel]].reshape( (x.shape[0], spe_width, 1, 2) ) for input_channel in range(1,6)])
    elif FLAGS.arch == 'multi_lstm':
        if FLAGS.lstm_order == 'ref_first':
            tmp_x = [np.concatenate([x[:,:,0].reshape((x.shape[0], 1, spe_width, 1, 1)), x[:,:,input_channel].reshape((x.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
        elif FLAGS.lstm_order == 'ref_last':
            tmp_x = [np.concatenate([x[:,:,input_channel].reshape((x.shape[0], 1, spe_width, 1, 1)), x[:,:,0].reshape((x.shape[0], 1, spe_width, 1, 1))], axis=1) for input_channel in range(1,6)]
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        y_=model.predict(tmp_x)
        y_ = y_[:,0,:,:,:]
    elif FLAGS.arch == 'lstm':
        tmp_x = []
        if FLAGS.lstm_order == 'ref_first':
            for i in range(6):
                tmp_x.append(x[:,:,i].reshape( (x.shape[0], 1, spe_width, 1, 1)))
        elif FLAGS.lstm_order == 'ref_last':
            for i in (1,2,3,4,5,0):
                tmp_x.append(x[:,:,i].reshape( (x.shape[0], 1, spe_width, 1, 1)))
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        tmp_x = np.concatenate(tmp_x, axis=1)
        y_=model.predict(tmp_x)
    elif FLAGS.arch == 'lstm_seq':
        tmp_x = []
        if FLAGS.lstm_order == 'ref_first':
            for i in range(6):
                tmp_x.append(x[:,:,i].reshape( (x.shape[0], 1, spe_width, 1, 1)))
        elif FLAGS.lstm_order == 'ref_last':
            for i in (1,2,3,4,5,0):
                tmp_x.append(x[:,:,i].reshape( (x.shape[0], 1, spe_width, 1, 1)))
        else:
            raise Exception('Unknown lstm order: {}'.format(FLAGS.lstm_order))
        tmp_x = np.concatenate(tmp_x, axis=1)
        y_=model.predict(tmp_x)
        y_=np.concatenate([y_[:,input_channel,:].reshape((y_.shape[0],spe_width,1,1)) for input_channel in range(1,6)], axis=-1)
    elif FLAGS.arch in ('unet6channel', 'unet6dim', 'unet6Pchannel', 'unet6Pdim', ):
        tmp_x = [x[:,:,input_channel].reshape( (x.shape[0], spe_width, 1, 1) ) for input_channel in range(6)]
        y_=model.predict(tmp_x)
    elif FLAGS.arch in ('unetD5channel', 'unetD5dim', 'unetD5Pchannel', 'unetD5Pdim', ):
        tmp_x = [(x[:,:,0]-x[:,:,input_channel]).reshape( (x.shape[0], spe_width, 1, 1) ) for input_channel in range(1,6)]
        y_=model.predict(tmp_x)
    end=time.time()
    print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 741 us per sample
    print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.1s per 100 samples
    
    
    # On va alors déterminer les métriques:
    threshold = .5
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
        
    
    # Calculons pour chaque pic réel/prédit la concordance
    # TODO : pour l'instant on regarde juste : quand un pic réel, qu'est-ce que c'est comme pic ?
    # A faire : regarder les pics uniquement prédits mais pas réels
    
    threshold=.5
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


    



    



    



    



    



    



    



    



    



    



    



