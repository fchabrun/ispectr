# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# Nouvelle approche 27 août 2021
# On va faire des conv 3D ->

# On va d'abord stacker dans une matrice de taille 304x5x2
# les courbes
# avec 304 points x 6 pistes
# et pour chaque piste 2 channels -> G/A/M/k/l + Ref/Ref/Ref/Ref/Ref/Ref
# ce qui fait que le model va se déplacer le long des 304 pts et des 2 courbes
# mais va à chaque fois interpréter les 2 channels (G/Ref, A/Ref, etc.)
# en même temps
# et donc reprérer ce qui fait un pic par les motifs G vs Ref, A vs Ref, etc.
# donc on aura une array X de dimensions : n x 304 x 5 x 2

print('Starting IF classification script...')

import argparse
import numpy as np
import tensorflow as tf
import pickle
import os
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--step", type=str, default='train')
parser.add_argument("--arch", type=str, default='3D')
parser.add_argument("--loss", type=str, default='binary_crossentropy')
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

# FLAGS for architecture hyperparameters
parser.add_argument("--blocks", type=int, default=4)
parser.add_argument("--wide_kernel_starts_at_block", type=int, default=3)
parser.add_argument("--kernel_size", type=int, default=8)
parser.add_argument("--filters", type=int, default=16)
parser.add_argument("--dropout", type=float, default=.1)
parser.add_argument("--batchnorm", type=int, default=1)

FLAGS = parser.parse_args()

debug = FLAGS.debug>0
host=FLAGS.host
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
do_part = FLAGS.part>0

blocks=FLAGS.blocks
wide_kernel_starts_at_block=FLAGS.wide_kernel_starts_at_block
kernel_size=FLAGS.kernel_size
filters=FLAGS.filters
dropout=FLAGS.dropout
batchnorm=FLAGS.batchnorm>0

model_name = 'seg3d-{}-l-{}-bs-{}-bl-{}-w-{}-k-{}-f{}-d-{}-bn-{}.h5'.format(FLAGS.arch,
                                                                            FLAGS.loss,
                                                                            BATCH_SIZE,
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
if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

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
    def __init__(self, add_fake_mspikes=None, add_fake_flcs=None, add_noise=None, add_shifting=None, do_not_cover=True, seed=42):
        super().__init__()
        self.add_fake_mspikes   = add_fake_mspikes
        self.add_fake_flcs      = add_fake_flcs
        self.add_noise          = add_noise
        self.add_shifting       = add_shifting
        self.do_not_cover       = do_not_cover # if set to true, won't add a m-spike at a location where a m-spike already exists
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
                        curve_with_mspike_y = ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1
                        if self.do_not_cover:
                            # check if covering an already existing m-spike
                            if np.any(np.logical_and(curve_with_mspike_y==1, batch_y[i,...].max(1)==1)):
                                continue
                        for dim in range(batch_x.shape[-1]):
                            if (dim==heavy_dim) | (dim==light_dim): # do not add! but add to y
                                batch_y[i,...,dim-1] = np.maximum(batch_y[i,...,dim-1], curve_with_mspike_y)
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
                        curve_with_mspike_y = ((curve_with_mspike)>(np.max(curve_with_mspike)/10))*1
                        if self.do_not_cover:
                            # check if covering an already existing m-spike
                            if np.any(np.logical_and(curve_with_mspike_y==1, batch_y[i,...].max(1)==1)):
                                continue
                        for dim in range(batch_x.shape[-1]):
                            if dim==light_dim: # do not add! but add to y
                                batch_y[i,...,dim-1] = np.maximum(batch_y[i,...,dim-1], curve_with_mspike_y)
                            else:
                                batch_x[i,...,dim] += curve_with_mspike + curve_with_mspike*np.random.normal(0,.01,curve_with_mspike.shape)
        return batch_x, batch_y
        
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=True, augmenter=None, seed=42, output="3D", allow_incomplete_batches = False, debug=False):
        self.x                  = x             # the x array
        self.y                  = y             # the y array
        self.batch_size         = batch_size    # the batch size
        self.shuffle            = shuffle       # set to true for shuffling samples
        self.augmenter          = augmenter
        self.seed               = seed
        self.output             = output
        self.rng                = np.random.RandomState(self.seed)
        self.debug              = debug
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
        batch_y = np.expand_dims(batch_y,-1)
        
        # return X and y
        if self.debug:
            batch_x_original = np.concatenate([np.expand_dims(np.tile(self.x[batch_indexes,...][...,[0]],5),-1), np.expand_dims(self.x[batch_indexes,...][...,1:],-1)], -1)
            # y do not need to be reshaped (only add 1 fake dim)
            batch_y_original = np.expand_dims(self.y[batch_indexes,...],-1)
            return batch_x, batch_y, batch_x_original, batch_y_original # return original arrays
        
        return batch_x, batch_y
    
# create our augmenter
standard_augmenter = ITAlgorithmicAugmenter(add_fake_mspikes=dict(freq=.2, # 20% fake mspikes
                                                                  minpos=180, # 180 is nice
                                                                  maxpos=251, # 251 is nice
                                                                  minheight=.5, # .5 is nice
                                                                  maxheight=8, # 8 is nice
                                                                  minwidth=3.5, # 3.5 is nice
                                                                  maxwidth=4.5, # 4.5 is nice
                                                                  mincount=1,
                                                                  maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                                            # add_fake_flcs=None,
                                            add_fake_flcs=dict(freq=.05, # 5% mspikes
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
                                            seed = 0)

# create our train generator
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=FLAGS.batch_size, augmenter = standard_augmenter, seed = 0)
valid_gen = ITGenerator(x=x_valid, y=y_valid, batch_size=FLAGS.batch_size, augmenter = None, seed = 0, shuffle=False, allow_incomplete_batches=True)
test_gen = ITGenerator(x=x_test, y=y_test, batch_size=FLAGS.batch_size, augmenter = None, seed = 0, shuffle=False, allow_incomplete_batches=True)

if debug:
    # a function for making plots for debugging our augmentation function
    def plotBatchData(batch_data, fr=0, to=9999999):
        from matplotlib import pyplot as plt
        
        def plotXYData(batch_x, batch_y, batch_x_original=None, batch_y_original=None, fr=0, to=999999):
            cols = 3
            if batch_x_original is not None:
                cols = 6
            for i in np.arange(max(0,fr),min(batch_x.shape[0],to)):
                plt.figure(figsize=(10*cols//3,9))
                for j,track in enumerate(['G','A','M','k','l']):
                    class_map=batch_y[i,:,j,0]
                    plt.subplot(5,cols,cols*j+1)
                    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                        plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x[i,peak_start:peak_end,j,0], color="red")
                    plt.plot(np.arange(0,spe_width), batch_x[i,:,j,0], '-')
                    plt.text(0,0,"Ref (aug input)")
                    plt.subplot(5,cols,cols*j+2)
                    for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                        plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x[i,peak_start:peak_end,j,1], color="red")
                    plt.plot(np.arange(0,spe_width), batch_x[i,:,j,1], '-')
                    plt.text(0,0,track+" (aug input)")
                    plt.subplot(5,cols,cols*j+3)
                    plt.plot(np.arange(0,spe_width), batch_y[i,:,j,0], '-')
                    plt.ylim(0,1)
                    plt.text(0,0,track+" (aug pred)")
                    if batch_x_original is not None: # same, for raw curves (before augmentation)
                        class_map=batch_y_original[i,:,j,0]
                        plt.subplot(5,cols,cols*j+4)
                        for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                            plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x_original[i,peak_start:peak_end,j,0], color="red")
                        plt.plot(np.arange(0,spe_width), batch_x_original[i,:,j,0], '-')
                        plt.text(0,0,"Ref (raw input)")
                        plt.subplot(5,cols,cols*j+5)
                        for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
                            plt.fill_between(x=np.arange(peak_start,peak_end), y1=0, y2=batch_x_original[i,peak_start:peak_end,j,1], color="red")
                        plt.plot(np.arange(0,spe_width), batch_x_original[i,:,j,1], '-')
                        plt.text(0,0,track+" (raw input)")
                        plt.subplot(5,cols,cols*j+6)
                        plt.plot(np.arange(0,spe_width), batch_y_original[i,:,j,0], '-')
                        plt.ylim(0,1)
                        plt.text(0,0,track+" (raw pred)")
                plt.tight_layout()
                plt.show()
            
        if len(batch_data)==4:
            batch_x, batch_y, batch_x_original, batch_y_original = batch_data
            plotXYData(batch_x, batch_y, batch_x_original, batch_y_original, fr=fr, to=to)
        else:
            batch_x, batch_y = batch_data
            plotXYData(batch_x, batch_y, fr=fr, to=to)
        
    
    # plot debug
    batch_ind = 0 # which batch to compute
    
    # run aug + get unaugmented curves
    train_gen.debug = True
    batch_data = train_gen.__getitem__(batch_ind)
    train_gen.debug = False
    plotBatchData(batch_data, to=8)
    

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
        
    def get_unet_custom(input_signal, blocks=4, wide_kernel_starts_at_block=-1, n_filters=16, kernel_size=3, dropout=0.1, batchnorm=True, n_classes=5, contract=1, return_model=True, return_sigmoid=True):
        # contracting path
        conv_blocks = []
        
        # encoder
        last_pb = input_signal
        for b in range(blocks):
            kernel_size_height = 1
            if (wide_kernel_starts_at_block>-1) and (b>=wide_kernel_starts_at_block-1):
                kernel_size_height=5
            dropout_mult=1.
            if b==0:
                dropout_mult=.5
            print("Adding conv layer in block {} with kernel size ({},{})".format(b+1,kernel_size,kernel_size_height))
            cb = conv1d_block(last_pb, n_filters=n_filters*np.power(2,b), kernel_size_width=kernel_size, kernel_size_height=kernel_size_height, batchnorm=batchnorm)
            conv_blocks.append(cb)
            if b<(blocks-1):
                pb = tf.keras.layers.MaxPooling2D((2,1)) (cb)
                pb = tf.keras.layers.Dropout(dropout*dropout_mult)(pb)
                last_pb = pb
        
        # decoder
        for b in range(blocks-1):
            ub = tf.keras.layers.Conv2DTranspose(n_filters*np.power(2,blocks-b-2), (kernel_size,1), strides=(2,1), padding='same') (conv_blocks[-1])
            ub = tf.keras.layers.Concatenate() ([ub, conv_blocks[blocks-b-2]])
            ub = tf.keras.layers.Dropout(dropout)(ub)
            cb = conv1d_block(ub, n_filters=n_filters*np.power(2,blocks-b-2), kernel_size_width=kernel_size, batchnorm=batchnorm)
            conv_blocks.append(cb)
        
        outputs = tf.keras.layers.Conv2D(n_classes, (1,contract), activation='sigmoid') (conv_blocks[-1])
        if return_model:
            model = tf.keras.models.Model(inputs=[input_signal], outputs=[outputs])
            return model
        if return_sigmoid:
            return outputs
        return conv_blocks[-1]
    
    if FLAGS.arch=="3D":
        input_signal = tf.keras.layers.Input((spe_width, 5, 2), name='input_if')
        model = get_unet_custom(input_signal,
                                blocks=blocks,
                                wide_kernel_starts_at_block=wide_kernel_starts_at_block,
                                n_filters=filters,
                                kernel_size=kernel_size,
                                dropout=dropout,
                                batchnorm=batchnorm,
                                n_classes=1,
                                contract=1,
                                return_model=True,
                                return_sigmoid=True)
    else:
        assert False, 'Not coded yet'
    
    verbose = 2-(host!="jeanzay")*1
    
    # on crée un callback pour surveiller l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, verbose=verbose, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=min_lr, verbose=verbose),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False)
    ]
    
    model.compile(loss=FLAGS.loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_iou])
    
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
np.save(os.path.join(path_out, log_name[:-4]+"_test.npy"), test_preds_)
        


    



    



    



    



    



    



    



    



    



    



    



