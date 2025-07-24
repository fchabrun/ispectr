# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# Nouvelle approche octobre 2021
# on va simplement classifier les courbes

print('Starting IF classification script...')

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# setting seed

tf.random.set_seed(31)
np.random.seed(32)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--step", type=str, default='train')
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

# FLAGS for architecture hyperparameters
parser.add_argument("--blocks", type=int, default=5)
parser.add_argument("--wide_kernel_starts_at_block", type=int, default=4)
parser.add_argument("--kernel_size", type=int, default=4)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--dropout", type=float, default=.05)
parser.add_argument("--batchnorm", type=int, default=1)

FLAGS = parser.parse_args()

debug = (FLAGS.debug>0) & (FLAGS.host=="local")
host=FLAGS.host
base_lr = 1e-2
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
do_part = FLAGS.part>0

blocks=FLAGS.blocks
wide_kernel_starts_at_block=FLAGS.wide_kernel_starts_at_block
kernel_size=FLAGS.kernel_size
filters=FLAGS.filters
dropout=FLAGS.dropout
batchnorm=FLAGS.batchnorm>0

model_name = 'cls3d-bs-{}-bl-{}-w-{}-k-{}-f{}-d-{}-bn-{}.h5'.format(BATCH_SIZE,
                                                                    blocks,
                                                                    wide_kernel_starts_at_block,
                                                                    kernel_size,
                                                                    filters,
                                                                    dropout,
                                                                    batchnorm)

log_name = model_name[:-2]+"pkl"

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if host == "local":
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = 'C:/Users/admin/Documents/Capillarys/temp2021'
elif host == "jeanzay":
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/out'
    debug=False
    
# %%

# load data
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
if_y = pd.read_csv(os.path.join(path_in,'if_simple_y.csv'))

# convert y to one hot encoding
if_y = pd.get_dummies(if_y)
y_labels = if_y.columns.tolist()
if_y = np.array(if_y)

# Partitionnement
# if do_part:
if False:
    part_rng = np.random.RandomState(seed=42)
    
    train_part = part_rng.choice(a = np.arange(if_x.shape[0]), size = if_x.shape[0]//2, replace = False)
    test_part = part_rng.choice(a = np.setdiff1d(np.arange(if_x.shape[0]), train_part), size = (if_x.shape[0]-train_part.shape[0])//2, replace = False)
    valid_part = np.setdiff1d(np.setdiff1d(np.arange(if_x.shape[0]), train_part), test_part)
    
    np.save(os.path.join(path_in, 'train_part.npy'), train_part)
    np.save(os.path.join(path_in, 'valid_part.npy'), valid_part)
    np.save(os.path.join(path_in, 'test_part.npy'), test_part)
else:
    train_part = np.load(os.path.join(path_in, 'train_part.npy'))
    valid_part = np.load(os.path.join(path_in, 'valid_part.npy'))
    test_part = np.load(os.path.join(path_in, 'test_part.npy'))

# check no overlaps
assert np.intersect1d(train_part,valid_part).shape[0]==0, "Error in partitions"
assert np.intersect1d(train_part,test_part).shape[0]==0, "Error in partitions"
assert np.intersect1d(valid_part,test_part).shape[0]==0, "Error in partitions"

x_train = if_x[train_part,:]
y_train = if_y[train_part,:]
x_valid = if_x[valid_part,:]
y_valid = if_y[valid_part,:]
x_test = if_x[test_part,:]
y_test = if_y[test_part,:]

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
class ITAugmenter():
    def __init__(self):
        pass
    
    def augment(self, batch_x, batch_y):
        assert False, "This method (ITAugmenter.augment) should be implemented in child class"
        # return batch_x.copy(), batch_y.copy()
        return batch_x, batch_y

class ITAlgorithmicAugmenter(ITAugmenter):
    def __init__(self, add_noise=None, add_shifting=None, seed=42):
        super().__init__()
        self.add_noise          = add_noise
        self.add_shifting       = add_shifting
        self.rng                = np.random.RandomState(seed)
        
    def augment(self, batch_x, batch_y):
        # batch_x, batch_y = batch_x.copy(), batch_y.copy()
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
                    else: # right
                        batch_x[i,...] = np.concatenate([np.zeros_like(batch_x[i,-d:,...]),batch_x[i,:-d,...]], axis=0)
        return batch_x, batch_y
        
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=True, augmenter=None, seed=42, output="3D", allow_incomplete_batches = False):
        self.x                  = x             # the x array
        self.y                  = y             # the y array
        self.batch_size         = batch_size    # the batch size
        self.shuffle            = shuffle       # set to true for shuffling samples
        self.augmenter          = augmenter
        self.seed               = seed
        self.output             = output
        self.rng                = np.random.RandomState(self.seed)
        self.allow_incomplete_batches = allow_incomplete_batches
        self.on_epoch_end()

    def __len__(self): # should return the number of batches per epoch
        if self.allow_incomplete_batches:
            return int(np.ceil(self.x.shape[0]/self.batch_size))
        return self.x.shape[0]//self.batch_size

    def on_epoch_end(self): # shuffle samples
        self.indexes = np.arange(self.x.shape[0])
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        # get indexes for current batch
        b_start = index*self.batch_size
        b_end = np.minimum(((index+1)*self.batch_size), len(self.indexes))
        batch_indexes = self.indexes[b_start:b_end]
        # get data
        batch_x = self.x[batch_indexes,...].copy()
        batch_y = self.y[batch_indexes,...].copy()
        # augment id needed
        if self.augmenter is not None:
            batch_x, batch_y = self.augmenter.augment(batch_x, batch_y)
        
        assert self.output == "3D", "Generator not suited for output other than '3D'"
        
        # reshape in 3D
        # take reference curves, repeat them and stack them with GAMkl curves
        batch_x = np.concatenate([np.expand_dims(np.tile(batch_x[...,[0]],5),-1), np.expand_dims(batch_x[...,1:],-1)], -1)
        # y do not need to be reshaped (only add 1 fake dim)
        # batch_y = np.expand_dims(batch_y,-1)
        
        return batch_x, batch_y
    
# create our augmenter
standard_augmenter = ITAlgorithmicAugmenter(# add_noise=None,
                                            add_noise=dict(freq=.2, minstd=.001, maxstd=.002),
                                            # add_shifting=None,
                                            add_shifting=dict(freq=.8, min=1, max=4),
                                            seed = 0)

# create our train generator
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=FLAGS.batch_size, augmenter = standard_augmenter, seed = 0)
valid_gen = ITGenerator(x=x_valid, y=y_valid, batch_size=FLAGS.batch_size, augmenter = None, seed = 0, shuffle=False, allow_incomplete_batches=True)
test_gen = ITGenerator(x=x_test, y=y_test, batch_size=FLAGS.batch_size, augmenter = None, seed = 0, shuffle=False, allow_incomplete_batches=True)

if debug:
    # a function for making plots for debugging our augmentation function
    def plotXYData(batch_x, batch_simple_y, fr=0, to=999999):
        from matplotlib import pyplot as plt
        cols = 2
        for i in np.arange(max(0,fr),min(batch_x.shape[0],to)):
            plt.figure(figsize=(10*cols//3,9))
            for j,track in enumerate(['G','A','M','k','l']):
                class_label=y_labels[np.argmax(batch_simple_y[i,:])]
                plt.subplot(5,cols,cols*j+1)
                plt.title("Class: {}".format(class_label))
                # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                #     plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x[i,peak_start:peak_end,j,0], color="red")
                plt.plot(np.arange(0,spe_width), batch_x[i,:,j,0], '-')
                plt.text(0,0,"Ref (aug input)")
                plt.subplot(5,cols,cols*j+2)
                # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                #     plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x[i,peak_start:peak_end,j,1], color="red")
                plt.plot(np.arange(0,spe_width), batch_x[i,:,j,1], '-')
                plt.text(0,0,track+" (aug input)")
            plt.tight_layout()
            plt.show()
            
    def plotBatchData(batch_data, fr=0, to=9999999):
        batch_x, batch_simple_y = batch_data
        plotXYData(batch_x, batch_simple_y, fr=fr, to=to)
        
    # plot debug
    batch_ind = 1 # which batch to compute
    
    # run aug + get unaugmented curves
    batch_data = train_gen.__getitem__(batch_ind)
    # batch_data = valid_gen.__getitem__(batch_ind)
    plotBatchData(batch_data, to=8)

# %%

if FLAGS.step=="train":
    
    # enfin, on va pouvoir réaliser le modèle de prédiction
    # réalisé depuis : https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
    # modifié pour marcher en "3D"
    def conv1d_block(input_tensor, n_filters, kernel_size_width=3, kernel_size_height=1, batchnorm=True):
        # first layer
        #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size_width,kernel_size_height), kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        # second layer
        #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size_width,kernel_size_height), kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x
        
    def get_classnet_custom(input_signal, blocks=5, wide_kernel_starts_at_block=-1, n_filters=32, kernel_size=3, dropout=0.1, batchnorm=True, n_classes=8):
        # contracting path
        x = input_signal
        # x = tf.keras.layers.Conv2D(n_filters, (1,1), kernel_initializer="he_normal", activation="relu") (x)
        
        for b in range(blocks):
            if wide_kernel_starts_at_block==-1:
                kernel_size_height = 1
            elif b>=wide_kernel_starts_at_block-1:
                kernel_size_height = 5
            else:
                kernel_size_height = 1
            x = conv1d_block(x, n_filters=n_filters*np.power(2,b), kernel_size_width=kernel_size, kernel_size_height=kernel_size_height, batchnorm=batchnorm)
            if b<blocks-1:
                x = tf.keras.layers.MaxPooling2D((2,1)) (x)
                
        x = tf.keras.layers.GlobalMaxPooling2D() (x)
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax') (x)
        
        # x = tf.keras.layers.GlobalMaxPooling2D() (x)
        # outputs = tf.keras.layers.Conv2D(n_classes, 1, activation='sigmoid') (x)
        model = tf.keras.models.Model(inputs=[input_signal], outputs=[outputs])
        
        model.summary()
        
        return model
    
    input_signal = tf.keras.layers.Input((spe_width, 5, 2), name='input_if')
    model = get_classnet_custom(input_signal,
                            blocks=blocks,
                            wide_kernel_starts_at_block=wide_kernel_starts_at_block,
                            n_filters=filters,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            batchnorm=batchnorm,
                            n_classes=len(y_labels))
    
    verbose = 2-(host!="jeanzay")*1
    
    # on crée un callback pour surveiller l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=verbose, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=min_lr, verbose=verbose),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False)
    ]
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    print(model.summary())
    

# %%

if FLAGS.step == "train":

    N_EPOCHS = 1000

    print('Setting batch size to: '+str(BATCH_SIZE))
    print('Setting maximal number of epochs to: '+str(N_EPOCHS))
    
    results = model.fit(train_gen,
                        batch_size=BATCH_SIZE,
                        epochs=N_EPOCHS,
                        callbacks=callbacks,
                        verbose=verbose,
                        validation_data=valid_gen,
                        validation_steps = valid_gen.__len__())
    
    # enfin, on voudra sauvegarder les résultats
    # Save history
    with open(os.path.join(path_out,log_name), 'wb') as file_pi:
        pickle.dump(results.history, file_pi)
        
    # model.save(os.path.join(path_out,'working_model.h5'))
    # model.save_weights(os.path.join(path_out,'working_weights.h5'))
    
    # plt.plot(results.history['loss'])
    # plt.plot(results.history['val_loss'])
        
# inference on test set and save results
test_preds_ = model.predict(test_gen)
# np.save(os.path.join(path_out, log_name[:-4]+"_test.npy"), test_preds_)
        
pred_labels = [y_labels[v][12:] for v in np.argmax(test_preds_, axis=1)]
real_labels = [y_labels[v][12:] for v in np.argmax(y_test, axis=1)]

pd.crosstab(pd.Series(pred_labels), pd.Series(real_labels))

np.sum(np.array(pred_labels)==np.array(real_labels))/len(pred_labels)
# 92.9% accuracy on test set
    



    



    



    



    



    



    



    



    



    



    



