# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:31:53 2021

@author: Floris Chabrun and Xavier Dieu

################# TRANSFORMERS TEST FOR SPE CLASSIFICATION ####################

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
parser.add_argument("--model", type = str,
                    default='base',
                    help='which transformer model to use ')
parser.add_argument("--model_args", type = str,
                    default='vit_1',
                    help='model args')
parser.add_argument("--epochs", type = int,
                    default=4000,
                    help='how many epochs to train the model on')
parser.add_argument("--class_weight", type = bool,
                    default=False,
                    help='weighting class(es)')
parser.add_argument("--action", type = str,
                    default="train_vit",
                    help='train or test')



args = parser.parse_args()


"""=============================================================================
Library imports
============================================================================="""

print('Loading libraries ...')

# general imports
import numpy as np
import os
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from ezml.misc import save_fig
import pandas as pd
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
import tensorflow.keras.backend as K
import pickle
from scipy.stats import norm
#import time
#from tqdm import trange
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
    INPUT_PATH = "/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data"
    print('Using {} as input path'.format(INPUT_PATH))
    OUTPUT_PATH = "/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/xavier"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print('Using {} as output path'.format(OUTPUT_PATH))

# seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# normalize
normc = True

# spe_width
spe_width = 304

# =============================================================================
# # To plot pretty figures
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
# 
# # Where to save the figures
# IMAGES_PATH = os.path.join(OUTPUT_PATH, "images")
# os.makedirs(IMAGES_PATH, exist_ok=True)
# 
# =============================================================================


# config for which model args to use
if args.model_args == 'default':
    epochs = args.epochs
    batch_size = 32
    num_heads = 16  # Number of attention heads
    embed_dim = 6 # "embedding" size (the 6 curves)
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    num_layer = 1 # number of transformer layer to stack
    embed_curve = False

if args.model_args == 'custom1':
    epochs = args.epochs
    batch_size = 32
    num_heads = 8  # Number of attention heads
    embed_dim = 64 # "embedding" size (the 6 curves)
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    num_layer = 1 # number of transformer layer to stack
    embed_curve = True

if args.model_args == 'custom2':
    epochs = args.epochs
    batch_size = 32
    num_heads = 8  # Number of attention heads
    embed_dim = 6 # "embedding" size (the 6 curves)
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    num_layer = 1 # number of transformer layer to stack
    embed_curve = False

if args.model_args == 'custom3':
    epochs = args.epochs
    batch_size = 32
    num_heads = 8  # Number of attention heads
    embed_dim = 64 # "embedding" size (the 6 curves)
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    num_layer = 4 # number of transformer layer to stack
    embed_curve = True

if args.model_args == 'custom4':
    epochs = args.epochs
    batch_size = 32
    num_heads = 8  # Number of attention heads
    embed_dim = 32 # "embedding" size (the 6 curves)
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    num_layer = 4 # number of transformer layer to stack
    embed_curve = True

if args.model_args == 'vit_1':
    epochs = args.epochs
    batch_size = 32
    num_heads = 1  # Number of attention heads
    embed_dim = 2 # "embedding" size (the 6 curves)
    ff_dim = 8  # Hidden layer size in feed forward network inside transformer
    key_dim = 2
    num_layer = 6 # number of transformer layer to stack
    embed_curve = None
    patch_size = None

# defining positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
    
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        if max_dims % 2 == 1: max_dims += 1 # max_dims must be even
        p, i = np.meshgrid(np.arange(self.max_steps), np.arange(self.max_dims // 2))
        pos_emb = np.empty((1, self.max_steps, self.max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / self.max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / self.max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]

    def get_config(self):
    
            config = super().get_config().copy()
            config.update({
                'max_steps': self.max_steps,
                'max_dims': self.max_dims,
            })
            return config

# defining the custom learning rate scheduler like in original transformer 
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=args.epochs/10, **kwargs):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
    
            config = {
                'd_model': self.d_model.numpy(),
                'warmup_steps': self.warmup_steps
            }
            return config


# we need to add other dimensions to our curve (similar to word embedding)
class ExpandCurve(tf.keras.layers.Layer) :
    def __init__(self, d_model, **kwargs) :
        super(ExpandCurve, self).__init__()
        self.d_model = d_model
    
    def call(self, x) :        
        return tf.stack(list(x for _ in range(self.d_model)), axis=2)

    def get_config(self):
    
            config = super().get_config().copy()
            config.update({
                'd_model': self.d_model,
            })
            return config


# if we want to augment/similar to word embeding the curves 
# by stacking the original inputs with output of some conv layers
class Curve_Embedding(tf.keras.layers.Layer):
    def __init__(self, curve_dim, 
                 final_curve_dim, 
                 **kwargs):
        super(Curve_Embedding, self).__init__()
        self.curve_dim = curve_dim
        self.final_curve_dim = final_curve_dim
        self.filters = (self.final_curve_dim - self.curve_dim) / 2
        self.conv1d_dense1 = tf.keras.layers.Dense(self.filters, 
                                                  kernel_initializer="he_normal")
        self.conv1d_dense2 = tf.keras.layers.Dense(self.filters, 
                                                  kernel_initializer="he_normal")
        self.conv1d_3x1_1 = tf.keras.layers.Conv1D(self.filters, 
                                                 3, 
                                                 padding='same',
                                                 kernel_initializer="he_normal",
                                                 name='conv1d_3x1')
        self.conv1d_3x1_2 = tf.keras.layers.Conv1D(self.filters, 
                                                 3, 
                                                 padding='same',
                                                 kernel_initializer="he_normal",
                                                 name='conv1d_3x1_2')
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.activ1 = tf.keras.layers.Activation("relu")
        self.activ2 = tf.keras.layers.Activation("relu")
        self.activ3 = tf.keras.layers.Activation("relu")
        self.activ4 = tf.keras.layers.Activation("relu")

    def call(self, inputs, training):
        # conv branches
        conv1x1_output = self.conv1d_dense1(inputs)
        conv1x1_output = self.batchnorm1(conv1x1_output)
        conv1x1_output = self.activ1(conv1x1_output)
        conv1x1_output = self.conv1d_dense2(conv1x1_output)
        conv1x1_output = self.batchnorm1(conv1x1_output)
        conv1x1_output = self.activ2(conv1x1_output)

        conv3x1_output = self.conv1d_3x1_1(inputs)
        conv3x1_output = self.batchnorm3(conv3x1_output)
        conv3x1_output = self.activ3(conv3x1_output)
        conv3x1_output = self.conv1d_3x1_2(conv3x1_output)
        conv3x1_output = self.batchnorm4(conv3x1_output)
        conv3x1_output = self.activ4(conv3x1_output)

        return tf.keras.layers.concatenate([inputs, conv1x1_output, conv3x1_output],axis=2)
            
    def get_config(self):
    
            config = super().get_config().copy()
            config.update({
                'curve_dim': self.curve_dim,
                'final_curve_dim': self.final_curve_dim,
            })
            return config



# config for which model to use 
if args.model == 'base':
    
    # class to implement the base transformer building block
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
                                       kernel_initializer="he_normal",
                                       activation="gelu"),
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
        
    # base transformer building function    
    def transformer_model_with_kt(hp, 
                                  spe_width=spe_width, 
                                  epochs=epochs, 
                                  seed=42) :
        
        # init some hp values for model construction           
        d_model = hp.Choice('num_embed', 
                            [8, 16])
        num_heads = d_model # keeping same number of heads as embed for simplicity
        ff_dim = hp.Choice('num_neurons_in_ff', 
                           [128, 256, 512])
        num_stack_transformers = hp.Int('num_transformer_blocks', 
                                        min_value = 1, 
                                        max_value = 2, 
                                        step = 1,
                                        default = 1)
        dropout_rate = hp.Choice('dropout_rate', 
                            [0.0, 0.2])
        learning_rate = CustomSchedule(d_model)        
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        # building a base transformer model
        inputs = tf.keras.layers.Input(shape=(spe_width))
        x = ExpandCurve(d_model)(inputs)                
        x = PositionalEncoding(spe_width, d_model)(x)        
        for _ in range(num_stack_transformers) :
            x = Transformer_Block(d_model, num_heads, ff_dim, rate=dropout_rate, seed=seed)(x)        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)                
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        
        # instantiating model and compiling it 
        model = tf.keras.Model(inputs=inputs, outputs=outputs)    
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])        
        return model


    
if args.model == 'evolved':
    
    class Transformer_Block(tf.keras.layers.Layer):
        def __init__(self, embed_dim, 
                     num_heads, 
                     ff_dim, 
                     rate=0.1,
                     **kwargs):
            super(Transformer_Block, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.gated_dense = tf.keras.layers.Dense(embed_dim, activation="relu")
            self.gated_sigmoid = tf.keras.layers.Dense(embed_dim, activation=tf.nn.sigmoid)
            self.conv1d_dense = tf.keras.layers.Dense(ff_dim, activation="relu")
            self.conv1d_3x1 = tf.keras.layers.Conv1D(ff_dim/8, 
                                                     3, 
                                                     padding='same',
                                                     name='conv1d_3x1',
                                                     activation="relu")
            self.sep_conv1d_9x1 = tf.keras.layers.SeparableConv1D(embed_dim, 
                                                                  9, 
                                                                  padding='same',
                                                                  name='sepconv1d_9x1',
                                                                  activation="relu")
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, 
                                                          key_dim=embed_dim)
            self.ffn = tf.keras.Sequential(
                [tf.keras.layers.Dense(ff_dim, activation="relu"), 
                 tf.keras.layers.Dense(embed_dim)])
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm6 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.dropout3 = tf.keras.layers.Dropout(rate)
            self.dropout4 = tf.keras.layers.Dropout(rate)

        def call(self, hidden_state, training):
            # gated linear unit
            residual_state = hidden_state
            hidden_state = self.layernorm1(hidden_state)
            values = self.gated_dense(hidden_state)
            gates = self.gated_sigmoid(hidden_state)
            hidden_state = values * gates
            hidden_state = self.layernorm2(residual_state + hidden_state)
            # conv branches
            residual_state = hidden_state
            left_output = self.conv1d_dense(hidden_state)
            left_output = self.dropout1(left_output, training=training)
            right_output = self.conv1d_3x1(hidden_state)
            right_output = self.dropout2(right_output, training=training)
            right_output = tf.pad(right_output,
              [[0, 0], [0, 0], [0, self.ff_dim - tf.cast(self.ff_dim/8, dtype=tf.int32)]],
              constant_values=0.)            
            hidden_state = self.layernorm3(left_output + right_output)
            hidden_state = self.sep_conv1d_9x1(hidden_state)
            hidden_state = self.layernorm4(residual_state + hidden_state)           
            # MultiHead Attention
            residual_state = hidden_state
            hidden_state = self.att(hidden_state, hidden_state)
            hidden_state = self.dropout3(hidden_state, training=training)
            hidden_state = self.layernorm5(residual_state + hidden_state)
            residual_state = hidden_state
            hidden_state = self.ffn(hidden_state)
            hidden_state = self.dropout4(hidden_state, training=training)
            return self.layernorm6(residual_state + hidden_state)
                
        def get_config(self):
        
                config = super().get_config().copy()
                config.update({
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim,
                })
                return config
        
    # base transformer building function    
    def transformer_model_with_kt(hp, 
                                  spe_width=spe_width, 
                                  epochs=epochs, 
                                  seed=42) :
        
        # init some hp values for model construction           
        d_model = hp.Choice('num_embed', 
                            [2, 4, 8, 16])
        num_heads = d_model # keeping same number of heads as embed for simplicity
        ff_dim = hp.Choice('num_neurons_in_ff', 
                           [256, 512, 1024, 2048])
        num_stack_transformers = hp.Int('num_transformer_blocks', 
                                        min_value = 1, 
                                        max_value = 4, 
                                        step = 1,
                                        default = 1)
        dropout_rate = hp.Float('dropout',
                        min_value=0.0,
                        max_value=0.5,
                        default=0.1,
                        step=0.1)
        learning_rate = CustomSchedule(d_model)        
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        # building a base transformer model
        inputs = tf.keras.layers.Input(shape=(spe_width))
        x = ExpandCurve(d_model)(inputs)                
        x = PositionalEncoding(spe_width, d_model)(x)        
        for _ in range(num_stack_transformers) :
            x = Transformer_Block(d_model, num_heads, ff_dim, rate=dropout_rate, seed=seed)(x)        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)                
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        
        # instantiating model and compiling it 
        model = tf.keras.Model(inputs=inputs, outputs=outputs)    
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        
        return model


if args.model == 'cross':
    
    # class to implement the base transformer building block
    class Transformer_Block(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, key_dim, rate=0.05, **kwargs):
            super(Transformer_Block, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.key_dim = key_dim
            self.rate = rate
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                          key_dim=self.key_dim)
            self.ffn = tf.keras.Sequential(
                [tf.keras.layers.Dense(self.ff_dim, activation="relu",
                                       kernel_initializer="he_normal"), 
                 tf.keras.layers.Dense(self.embed_dim, 
                                       kernel_initializer="he_normal")]
            )
            self.layernorm0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm00 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(self.rate)
            self.dropout2 = tf.keras.layers.Dropout(self.rate)
    
        def call(self, inputs, inputs2, training):
            inputs = self.layernorm0(inputs)
            inputs2 = self.layernorm00(inputs2)
            attn_output = self.att(inputs, inputs2)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

        def get_config(self):
        
                config = super().get_config().copy()
                config.update({
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim,
                    'rate': self.rate,
                })
                return config



# Generator for data augmentation
class ITGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size = 8, shuffle=False, add_fake_mspikes=None, 
                 add_fake_flcs=None, add_noise=None, add_shifting=None, seed=42, 
                 ford5Pchannel=False, return_6inputs = False, return_stack1input = False):
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
        return batch_x, batch_y
    

    
def conv1d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
        x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, 
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        # second layer
        #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
        x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, 
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x
    
    
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

# args for callbacks
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
savedir = os.path.join(OUTPUT_PATH,'run-{}'.format(now),'model.h5')
logdir = os.path.join(OUTPUT_PATH,'run-{}'.format(now))
mode = 'min'
monitor = 'val_loss'
patience = 20
#reduce_lr = int(round(epochs/18, 0))
min_delta = 0.0001


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
print('supervision set X shape: '+str(x_test.shape))
print('supervision set Y shape: '+str(y_test.shape))

# create our train generator
train_gen = ITGenerator(x=x_train, y=y_train, batch_size=batch_size,
                        # add_fake_mspikes=None,
                        add_fake_mspikes=dict(freq=.2,
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
                         #add_noise=None,
                        add_noise=dict(freq=0.1, minstd=.001, maxstd=.002),
                         #add_shifting=None,
                        add_shifting=dict(freq=.1, min=1, max=4),
                        seed = 0,
                        ford5Pchannel = False,
                        return_6inputs=False,
                        return_stack1input = True ) 

print('    ... data preprocessing done')



"""=============================================================================
TRAINING THE TRANSFORMER ON DATA 
============================================================================="""

# straight to model training 
if args.action == "train" :

    K.clear_session()

    inputs = tf.keras.layers.Input(shape=(spe_width, 6))
    if embed_curve :
        x = Curve_Embedding(6, embed_dim)(inputs)
        x = PositionalEncoding(spe_width, embed_dim)(x)                
        x_base = x        
    else :
        x = PositionalEncoding(spe_width, embed_dim)(inputs)
        x_base = x        
    for layer in range(num_layer) :
        x = Transformer_Block(embed_dim, num_heads, ff_dim)(x)
    if True :
        x = tf.keras.layers.Concatenate() ([x, x_base])
        x = tf.keras.layers.Dropout(rate=0.05)(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    outputs = tf.keras.layers.Conv1D(5, 1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    learning_rate = CustomSchedule(embed_dim)        
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model.compile(optimizer, loss="binary_crossentropy", metrics=[curve_iou])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=savedir, monitor=monitor, 
                                 save_best_only=True, mode=mode),
                 #tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, 
                               patience=patience, verbose=1, mode=mode, 
                               restore_best_weights=True),]
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, 
                 #                                     patience=reduce_lr, verbose=1, 
                 #                                     mode=mode, min_delta=min_delta, 
                 #                                     min_lr=0)]

    history = model.fit(x_train, y_train, # train_gen
                        batch_size=batch_size, 
                        epochs=epochs,                        
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=2)

    with open(os.path.join(OUTPUT_PATH, 'trainHistoryDict'+now+'.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



# or hyperparameter tuning with Keras Tuner
elif args.action == "train_kt" :

    K.clear_session()
    
    # setting Keras Tuner Hyperband hyperparameters search
    tuner = kt.Hyperband(
        transformer_model_with_kt,
        objective='val_loss',
        max_epochs=500,
        directory=OUTPUT_PATH,
        project_name='base_transformer_run1',
        seed=42)
    

    # implementing early stopping
    callbacks = [#ModelCheckpoint(filepath=savedir, monitor='val_loss', save_best_only=True, mode='min'),
                 #TensorBoard(log_dir=logdir),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=1, mode='min', restore_best_weights=True),
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, mode='min', min_delta=0.0001, min_lr=0)
                 ]


    # Weighting classes (for imbalanced data if submitted) ## NOT WORKING YET
    if args.class_weight :
        class_weight = dict()
        for num_class in range(len(pd.value_counts(y_train.shape[1]))) :
            class_weight[int(num_class)] = (1 / (np.unique(y_train, return_counts=True)[1][num_class]))*(len(y_train))/2.0
    else :
        class_weight = None
    
    # fitting model with KT hyperparameters search
    history = tuner.search(x_train,
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(x_super, y_super),
                        class_weight = class_weight,
                        callbacks = callbacks,
                        verbose=2
                            )


    # retrieving best model found by KT
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # saving best hyperparameters found
    hp_df = pd.DataFrame.from_dict(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values,
                                   orient='index')
    hp_df.to_csv(os.path.join(OUTPUT_PATH,'best_hp_'+args.model_args+now+'.csv'))
    
    # saving best model
    best_model.save(os.path.join(OUTPUT_PATH, 'best_'+args.model_args+now))
    


elif args.action == "train_tunet" :
    
    dropout_rate = 0.1
    
    K.clear_session()

    inputs_n = tf.keras.layers.Input(shape=(spe_width, 1))
    inputs_g = tf.keras.layers.Input(shape=(spe_width, 1))
    inputs_a = tf.keras.layers.Input(shape=(spe_width, 1))
    inputs_m = tf.keras.layers.Input(shape=(spe_width, 1))
    inputs_k = tf.keras.layers.Input(shape=(spe_width, 1))
    inputs_l = tf.keras.layers.Input(shape=(spe_width, 1))
    
    # encoder layer 1
    
    # normal curves
    output_n1 = conv1d_block(inputs_n, 32)    
    output_n2 = tf.keras.layers.MaxPool1D() (output_n1)   
    output_n2 = tf.keras.layers.Dropout(dropout_rate)  (output_n2)  

    output_n1 = PositionalEncoding(spe_width, embed_dim)(output_n1)
    
    # gamma curves
    output_g1 = conv1d_block(inputs_g, 32)    
    output_g2 = tf.keras.layers.MaxPool1D() (output_g1)   
    output_g2 = tf.keras.layers.Dropout(dropout_rate)  (output_g2)  
    
    output_g1 = PositionalEncoding(spe_width, embed_dim)(output_g1)
    output_g1_t = Transformer_Block(32, 8, 32*4, 4)(output_g1, output_n1)

    # alpha curves
    output_a1 = conv1d_block(inputs_a, 32)    
    output_a2 = tf.keras.layers.MaxPool1D() (output_a1)   
    output_a2 = tf.keras.layers.Dropout(dropout_rate) (output_a2) 
    
    output_a1 = PositionalEncoding(spe_width, embed_dim)(output_a1)
    output_a1_t = Transformer_Block(32, 8, 32*4, 4)(output_a1, output_n1)

    # mu curves
    output_m1 = conv1d_block(inputs_m, 32)    
    output_m2 = tf.keras.layers.MaxPool1D() (output_m1)   
    output_m2 = tf.keras.layers.Dropout(dropout_rate) (output_m2) 
    
    output_m1 = PositionalEncoding(spe_width, embed_dim)(output_m1)
    output_m1_t = Transformer_Block(32, 8, 32*4, 4)(output_m1, output_n1)

    # kappa curves
    output_k1 = conv1d_block(inputs_k, 32)    
    output_k2 = tf.keras.layers.MaxPool1D() (output_k1)   
    output_k2 = tf.keras.layers.Dropout(dropout_rate) (output_k2) 
    
    output_k1 = PositionalEncoding(spe_width, embed_dim)(output_k1)
    output_k1_t = Transformer_Block(32, 8, 32*4, 4)(output_k1, output_n1)

    # lambda curves
    output_l1 = conv1d_block(inputs_l, 32)    
    output_l2 = tf.keras.layers.MaxPool1D() (output_l1)   
    output_l2 = tf.keras.layers.Dropout(dropout_rate) (output_l2) 
    
    output_l1 = PositionalEncoding(spe_width, embed_dim)(output_l1)
    output_l1_t = Transformer_Block(32, 8, 32*4, 4)(output_l1, output_n1)
    

    # encoder layer 2
    
    # normal curves
    output_n3 = conv1d_block(output_n2, 64)    
    output_n4 = tf.keras.layers.MaxPool1D() (output_n3)   
    output_n4 = tf.keras.layers.Dropout(dropout_rate)  (output_n4)  
    
    output_n3 = tf.keras.layers.Conv1D(32, 1) (output_n3)
    output_n3 = PositionalEncoding(spe_width, embed_dim)(output_n3)
    
    # gamma curves
    output_g3 = conv1d_block(output_g2, 64)    
    output_g4 = tf.keras.layers.MaxPool1D() (output_g3)   
    output_g4 = tf.keras.layers.Dropout(dropout_rate)  (output_g4)  

    output_g3 = tf.keras.layers.Conv1D(32, 1) (output_g3)        
    output_g3 = PositionalEncoding(spe_width, embed_dim)(output_g3)
    output_g3_t = Transformer_Block(32, 8, 32*4, 4)(output_g3, output_n3)

    # alpha curves
    output_a3 = conv1d_block(output_a2, 64)    
    output_a4 = tf.keras.layers.MaxPool1D() (output_a3)   
    output_a4 = tf.keras.layers.Dropout(dropout_rate)  (output_a4)  
    
    output_a3 = tf.keras.layers.Conv1D(32, 1) (output_a3)        
    output_a3 = PositionalEncoding(spe_width, embed_dim)(output_a3)
    output_a3_t = Transformer_Block(32, 8, 32*4, 4)(output_a3, output_n3)

    # mu curves
    output_m3 = conv1d_block(output_m2, 64)    
    output_m4 = tf.keras.layers.MaxPool1D() (output_m3)   
    output_m4 = tf.keras.layers.Dropout(dropout_rate)  (output_m4)  
    
    output_m3 = tf.keras.layers.Conv1D(32, 1) (output_m3)        
    output_m3 = PositionalEncoding(spe_width, embed_dim)(output_m3)
    output_m3_t = Transformer_Block(32, 8, 32*4, 4)(output_m3, output_n3)

    # kappa curves
    output_k3 = conv1d_block(output_k2, 64)    
    output_k4 = tf.keras.layers.MaxPool1D() (output_k3)   
    output_k4 = tf.keras.layers.Dropout(dropout_rate)  (output_k4)  
    
    output_k3 = tf.keras.layers.Conv1D(32, 1) (output_k3)        
    output_k3 = PositionalEncoding(spe_width, embed_dim)(output_k3)
    output_k3_t = Transformer_Block(32, 8, 32*4, 4)(output_k3, output_n3)

    # lambda curves
    output_l3 = conv1d_block(output_l2, 64)    
    output_l4 = tf.keras.layers.MaxPool1D() (output_l3)   
    output_l4 = tf.keras.layers.Dropout(dropout_rate)  (output_l4)  
    
    output_l3 = tf.keras.layers.Conv1D(32, 1) (output_l3)        
    output_l3 = PositionalEncoding(spe_width, embed_dim)(output_l3)
    output_l3_t = Transformer_Block(32, 8, 32*4, 4)(output_l3, output_n3)
    

    # encoder layer 3
    
    # normal curves
    output_n5 = conv1d_block(output_n4, 128)    
    output_n6 = tf.keras.layers.MaxPool1D() (output_n5)   
    output_n6 = tf.keras.layers.Dropout(dropout_rate)  (output_n6)  

    output_n5 = tf.keras.layers.Conv1D(32, 1) (output_n5)
    output_n5 = PositionalEncoding(spe_width, embed_dim)(output_n5)
    
    # gamma curves
    output_g5 = conv1d_block(output_g4, 128)    
    output_g6 = tf.keras.layers.MaxPool1D() (output_g5)   
    output_g6 = tf.keras.layers.Dropout(dropout_rate)  (output_g6)  
    
    output_g5 = tf.keras.layers.Conv1D(32, 1) (output_g5)        
    output_g5 = PositionalEncoding(spe_width, embed_dim)(output_g5)
    output_g5_t = Transformer_Block(32, 8, 32*4, 4)(output_g5, output_n5)

    # alpha curves
    output_a5 = conv1d_block(output_a4, 128)    
    output_a6 = tf.keras.layers.MaxPool1D() (output_a5)   
    output_a6 = tf.keras.layers.Dropout(dropout_rate)  (output_a6)  
    
    output_a5 = tf.keras.layers.Conv1D(32, 1) (output_a5)        
    output_a5 = PositionalEncoding(spe_width, embed_dim)(output_a5)
    output_a5_t = Transformer_Block(32, 8, 32*4, 4)(output_a5, output_n5)

    # mu curves
    output_m5 = conv1d_block(output_m4, 128)    
    output_m6 = tf.keras.layers.MaxPool1D() (output_m5)   
    output_m6 = tf.keras.layers.Dropout(dropout_rate)  (output_m6)  
    
    output_m5 = tf.keras.layers.Conv1D(32, 1) (output_m5)        
    output_m5 = PositionalEncoding(spe_width, embed_dim)(output_m5)
    output_m5_t = Transformer_Block(32, 8, 32*4, 4)(output_m5, output_n5)

    # kappa curves
    output_k5 = conv1d_block(output_k4, 128)    
    output_k6 = tf.keras.layers.MaxPool1D() (output_k5)   
    output_k6 = tf.keras.layers.Dropout(dropout_rate)  (output_k6)  
    
    output_k5 = tf.keras.layers.Conv1D(32, 1) (output_k5)        
    output_k5 = PositionalEncoding(spe_width, embed_dim)(output_k5)
    output_k5_t = Transformer_Block(32, 8, 32*4, 4)(output_k5, output_n5)

    # lambda curves
    output_l5 = conv1d_block(output_k4, 128)    
    output_l6 = tf.keras.layers.MaxPool1D() (output_l5)   
    output_l6 = tf.keras.layers.Dropout(dropout_rate)  (output_l6)  
    
    output_l5 = tf.keras.layers.Conv1D(32, 1) (output_l5)        
    output_l5 = PositionalEncoding(spe_width, embed_dim)(output_l5)
    output_l5_t = Transformer_Block(32, 8, 32*4, 4)(output_l5, output_n5)


    # encoder layer 4
    
    # normal curves
    output_n7 = conv1d_block(output_n6, 256)    
    
    # gamma curves
    output_g7 = conv1d_block(output_g6, 256)    

    # alpha curves
    output_a7 = conv1d_block(output_a6, 256)    

    # mu curves
    output_m7 = conv1d_block(output_m6, 256)    

    # kappa curves
    output_k7 = conv1d_block(output_k6, 256)    

    # lambda curves
    output_l7 = conv1d_block(output_l6, 256)    
    
    
    # decoder portion

    # decoder layer 3
    output_ = tf.keras.layers.Concatenate() ([output_n7, output_g7, output_a7, output_m7, output_k7, output_l7])
    output_ = tf.keras.layers.Conv1D(256, 1) (output_)
    output_ = tf.keras.layers.Conv1DTranspose(128, 3, strides=2, padding='same') (output_)
    output_ = tf.keras.layers.Concatenate() ([output_, output_g5_t, output_a5_t,
                                              output_m5_t, output_k5_t,output_l5_t])
    output_ = tf.keras.layers.Dropout(dropout_rate)(output_)
    output_ = tf.keras.layers.Conv1D(256, 1) (output_)
    output_ = conv1d_block(output_, 128)

    # decoder layer 2
    output_ = tf.keras.layers.Conv1DTranspose(64, 3, strides=2, padding='same') (output_)
    output_ = tf.keras.layers.Concatenate() ([output_, output_g3_t, output_a3_t,
                                              output_m3_t, output_k3_t,output_l3_t])
    output_ = tf.keras.layers.Dropout(dropout_rate)(output_)
    output_ = tf.keras.layers.Conv1D(128, 1) (output_)
    output_ = conv1d_block(output_, 64)
    
    # decoder layer 1
    output_ = tf.keras.layers.Conv1DTranspose(32, 3, strides=2, padding='same') (output_)
    output_ = tf.keras.layers.Concatenate() ([output_, output_g1_t, output_a1_t,
                                              output_m1_t, output_k1_t,output_l1_t])
    output_ = tf.keras.layers.Dropout(dropout_rate)(output_)
    output_ = tf.keras.layers.Conv1D(64, 1) (output_)
    output_ = conv1d_block(output_, 32)
    
    output_ = tf.keras.layers.Conv1D(5, 1, activation="sigmoid")(output_)

        
    model = tf.keras.Model(inputs=[inputs_n,
                                   inputs_g,
                                   inputs_a,
                                   inputs_m,
                                   inputs_k,
                                   inputs_l], 
                           outputs=output_)

    learning_rate = CustomSchedule(embed_dim)        
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model.compile(optimizer, loss="binary_crossentropy", metrics=[curve_iou])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=savedir, monitor=monitor, 
                                 save_best_only=True, mode=mode),
                 #tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, 
                               patience=patience, verbose=1, mode=mode, 
                               restore_best_weights=True),]
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, 
                 #                                     patience=reduce_lr, verbose=1, 
                 #                                     mode=mode, min_delta=min_delta, 
                 #                                     min_lr=0)]



    n_train = x_train[:,:,0]
    g_train = x_train[:,:,1]
    a_train = x_train[:,:,2]
    m_train = x_train[:,:,3]
    k_train = x_train[:,:,4]
    l_train = x_train[:,:,5]

    n_valid = x_valid[:,:,0]
    g_valid = x_valid[:,:,1]
    a_valid = x_valid[:,:,2]
    m_valid = x_valid[:,:,3]
    k_valid = x_valid[:,:,4]
    l_valid = x_valid[:,:,5]
    
    history = model.fit(#[n_train, g_train, a_train, m_train, k_train, l_train], y_train, # train_gen
                        train_gen,
                        batch_size=batch_size, 
                        epochs=epochs,                        
                        validation_data=([n_valid, g_valid, a_valid, m_valid, k_valid, l_valid], y_valid),
                        callbacks=callbacks,
                        verbose=2)

    with open(os.path.join(OUTPUT_PATH, 'trainHistoryDict'+now+'.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)




elif args.action == "train_vit" :

    K.clear_session()
    
    inputs = tf.keras.layers.Input(shape=(spe_width*6))
    x = ExpandCurve(embed_dim)(inputs)
    x = PositionalEncoding(spe_width*6, embed_dim)(x)
    x_residual = x        
    for layer in range(num_layer) :
        x = Transformer_Block(embed_dim, num_heads, key_dim, ff_dim)(x)
    #x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    #x_residual = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x_residual)
    x = tf.keras.layers.Lambda(lambda x : x[:,304:])(x)
    #x = tf.keras.layers.Reshape((304,5))(x)
    x_residual = tf.keras.layers.Lambda(lambda x : x[:,304:])(x_residual)
    #x_residual = tf.keras.layers.Reshape((304,5))(x_residual)
    x = tf.keras.layers.Concatenate() ([x, x_residual])
    outputs = tf.keras.layers.Conv1D(1, 1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    learning_rate = CustomSchedule(1024)        
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model.compile(optimizer, loss="binary_crossentropy", metrics=[curve_iou])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=savedir, monitor=monitor, 
                                 save_best_only=True, mode=mode),
                 #tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, 
                               patience=patience, verbose=1, mode=mode, 
                               restore_best_weights=True),]
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, 
                 #                                     patience=reduce_lr, verbose=1, 
                 #                                     mode=mode, min_delta=min_delta, 
                 #                                     min_lr=0)]

    # reshaping input to stack all 6 curves after each other
    #x_train = np.hstack([x_train[:,:,curve] for curve in range(6)])    
    x_valid = np.hstack([x_valid[:,:,curve] for curve in range(6)])    
    #y_train = np.hstack([y_train[:,:,curve] for curve in range(5)])    
    y_valid = np.hstack([y_valid[:,:,curve] for curve in range(5)])    
    

    history = model.fit(train_gen, # x_train, y_train
                        batch_size=batch_size, 
                        epochs=epochs,                        
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=2)

    with open(os.path.join(OUTPUT_PATH, 'trainHistoryDict'+now+'.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)




elif args.action == "train_vitunet" :

    K.clear_session()

    inputs = tf.keras.layers.Input(shape=(spe_width*6))
    x = tf.keras.layers.Conv1D(embed_dim, patch_size, 
                               strides=patch_size,
                               kernel_initializer="he_normal") (inputs)
    x = PositionalEncoding(spe_width*6, embed_dim)(x)
    x_residual = x        
    for layer in range(num_layer) :
        x = Transformer_Block(embed_dim, num_heads, ff_dim)(x)
    if True :
        x = tf.keras.layers.Concatenate() ([x, x_base])
        x = tf.keras.layers.Dropout(rate=0.05)(x)
        x = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=3,
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=3,
                                   kernel_initializer="he_normal", padding="same", 
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    outputs = tf.keras.layers.Conv1D(5, 1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    learning_rate = CustomSchedule(embed_dim)        
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model.compile(optimizer, loss="binary_crossentropy", metrics=[curve_iou])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=savedir, monitor=monitor, 
                                 save_best_only=True, mode=mode),
                 #tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, 
                               patience=patience, verbose=1, mode=mode, 
                               restore_best_weights=True),]
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, 
                 #                                     patience=reduce_lr, verbose=1, 
                 #                                     mode=mode, min_delta=min_delta, 
                 #                                     min_lr=0)]

    history = model.fit(x_train, y_train, # train_gen
                        batch_size=batch_size, 
                        epochs=epochs,                        
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=2)

    with open(os.path.join(OUTPUT_PATH, 'trainHistoryDict'+now+'.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# for testing the lr schedule
# =============================================================================
# learning_rate = CustomSchedule(2)
# 
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
# 
# import matplotlib.pyplot as plt
# 
# temp_learning_rate_schedule = CustomSchedule(2)
# plt.plot(temp_learning_rate_schedule(tf.range(4000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# 
# 
# =============================================================================
# TEST for reloading model

# =============================================================================
# from tensorflow.keras.models import load_model
# model = load_model(r"D:\Anaconda datasets\Capillarys\transformer_class\output\run-20210423115627\model.h5",
#                    custom_objects={'ExpandCurve':ExpandCurve,
#                                    'Transformer_Block':Transformer_Block,
#                                    'CustomSchedule':CustomSchedule,
#                                    'PositionalEncoding':PositionalEncoding})
# model.summary()
# print(model.predict(x_super[100:200,:]).shape)
# model.evaluate(x_super[100:200,:], y_super[100:200,:] )
# 
# # WORKS
# 
# hist_path = r"D:\Anaconda datasets\Capillarys\transformer_class\output\trainHistoryDict20210422144919.pkl"
# hist_tmp = pickle.load(open(hist_path, 'rb'))
# 
# 
# =============================================================================



# Validation
if False:
    
    # predict
    export_metrics = dict()
    
    model = tf.keras.models.load_model(r"D:\Anaconda datasets\Capillarys\IF_transformer\output\run-20210429125743\model.h5",
                                       compile=False,
                                       custom_objects={'Curve_Embedding':Curve_Embedding,
                                                       'Transformer_Block': Transformer_Block,
                                                       'PositionalEncoding': PositionalEncoding,
                                                       'CustomSchedule':CustomSchedule})
    
    use_set = 'test'
    if use_set=='train':
        x = x_train
        y = y_train
    elif use_set=='valid':
        x = x_valid
        y = y_valid
    elif use_set=='test':
        n_test = x_test[:,:,0]
        g_test = x_test[:,:,1]
        a_test = x_test[:,:,2]
        m_test = x_test[:,:,3]
        k_test = x_test[:,:,4]
        l_test = x_test[:,:,5]

        x = [n_test, g_test, a_test, m_test, k_test, l_test]
        y = y_test
    
    size = n_test.shape[0]
    start=time.time()
    y_ = model.predict(x)
    end=time.time()
    print("Time per sample: "+format(round(1000*(end-start)/size, 3))+'ms') # 741 us per sample
    print("Time for 100 samples: "+format(round(100*(end-start)/size, 1))+'s') # 0.1s per 100 samples
    
    y_.shape
    y_ = np.expand_dims(y_, axis=2)
    
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
    
    if False:
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
    
    threshold=0.05 # ou 0.5
    curve_ids = []
    groundtruth_spikes = []
    predicted_spikes = []
    for ix in trange(n_test.shape[0]):
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
    
        
    print('Global precision: '+str(round(100*np.sum(conc_df.true==conc_df.pred)/conc_df.shape[0],1)))
    for typ in np.unique(conc_df.true):
        subset=conc_df.true==typ
        print('  Precision for type '+typ+': '+str(round(100*np.sum(conc_df.true.loc[subset]==conc_df.pred.loc[subset])/np.sum(subset), 1)))
        export_metrics['Acc-{}'.format(typ)] = 100*np.sum(conc_df.true.loc[subset]==conc_df.pred.loc[subset])/np.sum(subset)
    export_metrics['Acc-global'] = 100*np.sum(conc_df.true==conc_df.pred)/conc_df.shape[0]
    
    export_metrics['Mistakes-total'] = conc_df.loc[conc_df.true!=conc_df.pred,:].shape[0]
    mistakes = conc_df.loc[conc_df.true!=conc_df.pred,'ix'].unique().tolist()
    export_metrics['Mistakes-curves'] = len(mistakes)
    
    model.compile(optimizer, loss="binary_crossentropy", metrics=[curve_iou])
    model.evaluate(x,y)
    
    
    # Quand il n'y a pas de pic, qu'est-ce que le modèle renvoie ??
    # -> faire une AUC
    if False:
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
    

