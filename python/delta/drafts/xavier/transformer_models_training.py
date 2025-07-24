# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:31:53 2021

@author: Floris Chabrun and Xavier Dieu

################# TRANSFORMERS TRAINING FOR IS ################################

"""

print('Lauching Script ...')


"""=============================================================================
Arguments with argparse
============================================================================="""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type = str,
                    default='local',
                    help='data path')
parser.add_argument("-s", "--seed", type = int,
                    default=42,
                    help='the random seed')
parser.add_argument("--epochs", type = int,
                    default=200,
                    help='how many epochs to train the model on')
parser.add_argument("--action", type = str,
                    default="search_models",
                    help='train or test')



args = parser.parse_args()


"""=============================================================================
Library imports
============================================================================="""

print('Loading libraries ...')

# general imports
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
import tensorflow.keras.backend as K
import pickle
from scipy.stats import norm
import itertools
import pandas as pd
print('    ...libraries loaded')


"""=============================================================================
CONFIG Settings
============================================================================="""

# PATHs
if args.path == 'local' :
    INPUT_PATH = r"D:\Anaconda datasets\Capillarys\IF_transformer"
    print('Using {} as input path'.format(INPUT_PATH))
    OUTPUT_PATH = r"D:\Anaconda datasets\Capillarys\IF_transformer\output"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print('Using {} as output path'.format(OUTPUT_PATH))


elif args.path == 'jeanzay' :
    INPUT_PATH = "/gpfsdswork/projects/rech/ild/uqk67mt/delta_trans/data"
    print('Using {} as input path'.format(INPUT_PATH))
    OUTPUT_PATH = "/gpfsdswork/projects/rech/ild/uqk67mt/delta_trans/out"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print('Using {} as output path'.format(OUTPUT_PATH))

# seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# normalize
normc = False # keep curves in 0-1 range

# spe_width
spe_width = 304

# base model args
epochs = args.epochs
batch_size = 8
num_heads = [1, 3]  # Number of attention heads
embed_dims = [24, 48, 96] # "embedding" size (the 6 curves, or more)
num_blocks = [6] # number of transformer block in each layer
num_filters = [256, 512] # number of filters of conv decoders
num_convblocks = [1, 2]
patch_size = 4

# base args for callbacks
mode = 'min'
monitor = "val_main_out_loss"
patience = 20
reduce_lr = 10  
min_delta = 0.0001



"""=============================================================================
Custom Classes 
============================================================================="""

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


# from efficient Net and adapted
def swish(x):
    return x * tf.nn.sigmoid(x)



class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.input_channels = input_channels
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.reduce_conv = tf.keras.layers.Conv1D(filters=self.num_reduced_filters,
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv1D(filters=input_channels,
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        #branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output

    def get_config(self):
    
        config = super().get_config().copy()
        config.update({
            'input_channels': self.input_channels,
        })
        return config


class MBConv(tf.keras.layers.Layer):
    def __init__(self, channels, drop_rate):
        super(MBConv, self).__init__()
        self.channels = channels
        self.drop_rate = drop_rate
        self.conv1 = tf.keras.layers.Conv1D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.SeparableConv1D(filters=channels,
                                                      kernel_size=3,
                                                      strides=1,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=channels)
        self.conv2 = tf.keras.layers.Conv1D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.drop_rate:
            x = self.dropout(x, training=training)
        x = tf.keras.layers.add([x, inputs])
        return x

    def get_config(self):
    
        config = super().get_config().copy()
        config.update({
            'channels': self.channels,
            'drop_rate': self.drop_rate,
        })
        return config


# Generator for data augmentation
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=False, add_fake_mspikes=None, 
                 add_fake_flcs=None, add_noise=None, add_shifting=None, seed=42, 
                 ford5Pchannel=False, return_6inputs = False, return_stack1input = False,
                 return_4y=False):
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
        self.ford5Pchannel      = ford5Pchannel # for special u-net architecture
        self.on_epoch_end()
        self.return_6inputs     = return_6inputs
        self.return_stack1input = return_stack1input
        self.return_4y          = return_4y
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
        
        if self.ford5Pchannel:
            return [np.expand_dims((batch_x[...,0]-batch_x[...,dim]), axis=(2,3)) for dim in range(1,batch_x.shape[-1])], np.expand_dims(batch_y, axis=2)
        
        if self.return_6inputs :
            return [batch_x[...,0], batch_x[...,1], batch_x[...,2], batch_x[...,3], batch_x[...,4], batch_x[...,5]], batch_y
        if self.return_stack1input :
            return np.hstack([batch_x[:,:,curve] for curve in range(6)]), np.hstack([batch_y[:,:,curve] for curve in range(5)])
        if self.return_4y :
            return batch_x, [batch_y, batch_y, batch_y] #batch_y
        return batch_x, batch_y
    
    
    
# custom metric
def curve_iou(y_true, y_pred, smooth = 1e-5):
    trh = tf.cast(tf.greater(y_true, .5), 'double')
    prd = tf.cast(tf.greater(y_pred, .5), 'double')
    i = tf.cast(tf.greater(trh+prd, 1), 'double')
    u = tf.cast(tf.greater(trh+prd, 0), 'double')
    i = tf.reduce_sum(i)
    u = tf.reduce_sum(u)
    return (smooth+i) / (smooth+u)

tf.keras.metrics.curve_iou=curve_iou



"""=============================================================================
LOADING AND PREPROCESSING DATA 
============================================================================="""

# load raw data
print('preprocessing raw data ...')
if_x = np.load(os.path.join(INPUT_PATH,'if_v1_x.npy'))
if_y = np.load(os.path.join(INPUT_PATH,'if_v1_y.npy'))
print('    ... raw data loaded')

# data splitting
train_part = np.load(os.path.join(INPUT_PATH, 'train_part.npy'))
valid_part = np.load(os.path.join(INPUT_PATH, 'valid_part.npy'))
test_part = np.load(os.path.join(INPUT_PATH, 'test_part.npy'))

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

print('    ... raw data splitted')


if normc:
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
print('test set X shape: '+str(x_test.shape))
print('test set Y shape: '+str(y_test.shape))

# create our train generator
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=batch_size,
                        # add_fake_mspikes=None,
                        add_fake_mspikes=dict(freq=.1,
                                              minpos=180, # 180 is nice
                                              maxpos=251, # 251 is nice
                                              minheight=.5, # .5 is nice
                                              maxheight=8, # 8 is nice
                                              minwidth=3.5, # 3.5 is nice
                                              maxwidth=4.5, # 4.5 is nice
                                              mincount=1,
                                              maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                        # add_fake_flcs=None,
                        add_fake_flcs=dict(freq=.02,
                                           minpos=100, # 180 is nice
                                           maxpos=251, # 251 is nice
                                           minheight=.4, # .5 is nice
                                           maxheight=.5, # 8 is nice
                                           minwidth=2.5, # 3.5 is nice
                                           maxwidth=3.5, # 4.5 is nice
                                           mincount=1,
                                           maxcount=1), # if more than 1, may be unpredictable (i.e. 2 spikes at same location...)
                         #add_noise=None,
                        add_noise=dict(freq=0.1, minstd=.001, maxstd=.002),
                         #add_shifting=None,
                        add_shifting=dict(freq=.1, min=1, max=4),
                        seed = 0,
                        ford5Pchannel = False,
                        return_6inputs=False,
                        return_stack1input = False,
                        return_4y=False) 
        
print('    ... data preprocessing done')



"""=============================================================================
TRAINING THE TRANSFORMER ON DATA 
============================================================================="""


# grid search to fine tune some hyperparameters
if args.action == 'search_models' :
    
    # first creating a list of all hyperparameters list to create all 
    # possible combinations 
    hparams_list = [embed_dims, num_blocks, num_heads, num_filters, num_convblocks]
    
    cols = ['HP', 
            'train_loss', 
            'val_loss',
            'test_loss',
            'train_iou',
            'val_iou',
            'test_iou',
            ]
    
    # init a results df    
    df = pd.DataFrame(index=range(len(list(itertools.product(*hparams_list)))), 
                      columns = cols)

    df_index = 0
    
    # we will test different number of transformers per layer
    for embed_dim, num_block, num_head, filters, num_convblock in itertools.product(*hparams_list) :  

        ff_dim = embed_dim*4  # Hidden layer size in feed forward network inside transformer
        key_dim = int(embed_dim/num_head)
            
        # creating the model's architecture
        # first making sure session is empty
        K.clear_session()
        
        # inputs the stacked 6 curves
        inputs = tf.keras.layers.Input(shape=(spe_width, 6))
        
        
        # encoder first layer
        x_e1 = tf.keras.layers.Conv1D(embed_dim, 
                                      7, 
                                      strides=4,
                                      padding='same') (inputs)
        for layer in range(num_block) :
            x_e1 = Transformer_Block(embed_dim, num_head, key_dim, ff_dim)(x_e1)
        
        # encoder second layer 
        x_e2 = tf.keras.layers.Conv1D(embed_dim*2, 
                                      3, 
                                      strides=2,
                                      padding='same') (x_e1)
        for layer in range(num_block) :
            x_e2 = Transformer_Block(embed_dim*2, num_head*2, key_dim, ff_dim*2)(x_e2)
    
        # encoder Third layer 
        x_e3 = tf.keras.layers.Conv1D(embed_dim*4, 
                                      3, 
                                      strides=2,
                                      padding='same') (x_e2)
        for layer in range(num_block) :
            x_e3 = Transformer_Block(embed_dim*4, num_head*4, key_dim, ff_dim*4)(x_e3)
    
    
        # decoder for 2nd layer
        x_e1_maxpool = tf.keras.layers.MaxPool1D()(x_e1) 
        x_e3t = tf.keras.layers.Conv1DTranspose(embed_dim*4, 
                                                2, 
                                                strides=2,
                                                padding="same") (x_e3)
        x_d2 = tf.keras.layers.Concatenate() ([x_e3t, x_e2, x_e1_maxpool])
        x_d2 = tf.keras.layers.Conv1D(filters=filters, 
                                  kernel_size=1,
                                  padding="same", 
                                  data_format='channels_last')(x_d2)
        x_d2 = swish(x_d2)
        for layer in range(num_convblock) :
            x_d2 = MBConv(filters, 0.1)(x_d2)
        
        # decoder for 1st layer
        x_d2t = tf.keras.layers.Conv1DTranspose(filters, 
                                                2, 
                                                strides=2,
                                                padding="same") (x_d2)
        x_e3t2 = tf.keras.layers.Conv1DTranspose(embed_dim*4, 
                                                 4, 
                                                 strides=4,
                                                 padding="same") (x_e3)
        x_d1 = tf.keras.layers.Concatenate() ([x_d2t, x_e1, x_e3t2])
        x_d1 = tf.keras.layers.Conv1D(filters=filters, 
                                  kernel_size=1,
                                  padding="same", 
                                  data_format='channels_last')(x_d1)
        x_d1 = swish(x_d1)
        for layer in range(num_convblock) :
            x_d1 = MBConv(filters, 0.1)(x_d1)
    
        # decoder/output for input layer
        x_d1t = tf.keras.layers.Conv1DTranspose(filters, 
                                                patch_size, 
                                                strides=patch_size,
                                                padding="same") (x_d1)
        x_d2t2 = tf.keras.layers.Conv1DTranspose(filters, 
                                                 patch_size*2, 
                                                 strides=patch_size*2,
                                                 padding="same") (x_d2)
        x_e3t3 = tf.keras.layers.Conv1DTranspose(embed_dim*4, 
                                                 patch_size*4, 
                                                 strides=patch_size*4,
                                                 padding="same") (x_e3)
        x_out = tf.keras.layers.Concatenate() ([inputs, x_d1t, x_d2t2, x_e3t3])
        x_out = tf.keras.layers.Conv1D(filters=filters, 
                                  kernel_size=1,
                                  padding="same", 
                                  data_format='channels_last')(x_out)
        x_out = swish(x_out)
        for layer in range(num_convblock) :
            x_out = MBConv(filters, 0.1)(x_out)
        x_out = tf.keras.layers.Conv1D(5, 
                                       1, 
                                       activation="sigmoid", 
                                       name='main_out')(x_out)
        
        
        # Deep Supervision
        # Decoder 2
        x_out_e2 = tf.keras.layers.Conv1D(filters=5, 
                                          kernel_size=3,
                                          padding="same", 
                                          data_format='channels_last')(x_d2)
        x_out_e2 = tf.keras.layers.BatchNormalization()(x_out_e2)
        x_out_e2 = tf.keras.layers.Conv1DTranspose(5, 
                                                   patch_size*2, 
                                                   strides=patch_size*2,
                                                   padding="same") (x_out_e2)    
        x_out_e2 = tf.keras.layers.Activation("sigmoid", 
                                              name='deep_out2')(x_out_e2)    
        
        # Decoder 1
        x_out_e1 = tf.keras.layers.Conv1D(filters=5, 
                                          kernel_size=3,
                                          padding="same", 
                                          data_format='channels_last')(x_d1)
        x_out_e1 = tf.keras.layers.BatchNormalization()(x_out_e1)
        x_out_e1 = tf.keras.layers.Conv1DTranspose(5, 
                                                   patch_size, 
                                                   strides=patch_size,
                                                   padding="same") (x_out_e1)    
        x_out_e1 = tf.keras.layers.Activation("sigmoid", 
                                              name='deep_out1')(x_out_e1)
        
        
        # Instantiating SiT4+ model
        model = tf.keras.Model(inputs=inputs, 
                               outputs=[x_out, x_out_e1, x_out_e2]) 
    
        optimizer = tf.keras.optimizers.Nadam()
    
        model.compile(optimizer, 
                      loss="binary_crossentropy", 
                      metrics=[curve_iou],
                      loss_weights=[0.6, 0.2, 0.2]) 
        
                
        # callbacks
        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        savedir = os.path.join(OUTPUT_PATH,'run-{}_e{}_b{}_h{}_f{}_c{}'.format(now, embed_dim, num_block, num_head, filters, num_convblock),'model.h5')
        logdir = os.path.join(OUTPUT_PATH,'run-{}_e{}_b{}_h{}_f{}_c{}'.format(now, embed_dim, num_block, num_head, filters, num_convblock))            
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=savedir, 
                                                        monitor=monitor, 
                                                        save_best_only=True, 
                                                        mode=mode),
                     tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                      min_delta=min_delta, 
                                                      patience=patience, 
                                                      verbose=1, 
                                                      mode=mode, 
                                                      restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, 
                                                          factor=0.1, 
                                                          patience=reduce_lr, 
                                                          verbose=1, 
                                                          mode=mode, 
                                                          min_delta=min_delta, 
                                                          min_lr=1e-5)]
       
        
        # fitting model             
        history = model.fit(train_gen,
                            batch_size=batch_size, 
                            epochs=epochs,                        
                            validation_data=(x_valid, [y_valid, y_valid, y_valid]), 
                            callbacks=callbacks,
                            verbose=2)
    
        with open(os.path.join(OUTPUT_PATH, 'trainHistoryDict'+now+'.pkl'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
                
        train_results = model.evaluate(x_train, [y_train, y_train, y_train])
        val_results = model.evaluate(x_valid, [y_valid, y_valid, y_valid])
        test_results = model.evaluate(x_test, [y_test, y_test, y_test])
        df.loc[df_index, 'HP'] = 'e{}_b{}_h{}_f{}_c{}'.format(embed_dim, num_block, num_head, filters, num_convblock)
        df.loc[df_index, 'train_loss'] = train_results[1]
        df.loc[df_index, 'val_loss'] = val_results[1]
        df.loc[df_index, 'test_loss'] = test_results[1]
        df.loc[df_index, 'train_iou'] = train_results[4]
        df.loc[df_index, 'val_iou'] = val_results[4]
        df.loc[df_index, 'test_iou'] = test_results[4]
        
        df_index += 1
        
        # df to csv
        df.to_csv(os.path.join(OUTPUT_PATH, 'results.csv'))
        
# training a model with the best hyperparameters and architecture found do far
if args.action == 'train_model' :
    pass # TO DO : train one model using the best architecture and hyperparameters
