# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:04:31 2021

@author: admin
"""

# architecture globale pensée :
# modèle G1 : ELP sans pic + position -> ELP avec pic
# modèle G2 : ELP avec pic + position ou 0 -> ELP avec pic gardé ou sans pic

# https://arxiv.org/pdf/1611.07004.pdf
# https://www.tensorflow.org/tutorials/generative/pix2pix

# ici on va faire en 1 fois:
# on génère direcetment les 5 courbes

import argparse
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
import re
plt.ioff()

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="local")
# parser.add_argument("--step", type=str, default='test')
parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--steps_per_epoch", type=int, default=100)


parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--n_filters", type=int, default=32)
parser.add_argument("--n_layers_per_block", type=int, default=1)

parser.add_argument("--learning_rate", type=float, default=1e-04)
parser.add_argument("--loss", type=str, default="basic", help='loss used, either "basic" or "wasserstein"')

FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if FLAGS.host=="local":
    path_nospike_elp = r"C:\Users\admin\Documents\Capillarys\data\delta_gan\data\spe_elp_matrix_v0.csv"
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = r'C:\Users\admin\Documents\Capillarys\data\delta_gan\out'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/out'
    
DEBUG = FLAGS.host=="local"
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

spe_width = 304

# first load : convert elp_dataset to numpy array for quicker loading
if False:
    elp_dataset = pd.read_csv(path_nospike_elp)
    elp_nomspike_x = elp_dataset.loc[elp_dataset.loc[:,"class"]==0,[s for s in elp_dataset.columns if re.match("^elp[0-9]+$", s) is not None]]
    elp_nomspike_x = np.array(elp_nomspike_x)
    elp_nomspike_x = (elp_nomspike_x / np.expand_dims(elp_nomspike_x.max(axis=1), axis=-1))
    np.save(os.path.join(path_in, 'elpspe_v1_x.npy'), elp_nomspike_x)
    
    spe_nomspike_x = elp_dataset.loc[elp_dataset.loc[:,"class"]==0,[s for s in elp_dataset.columns if re.match("^spe[0-9]+$", s) is not None]]
    spe_nomspike_x = np.array(spe_nomspike_x)
    
    # i=1
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(np.arange(spe_width), elp_nomspike_x[i,...])
    # plt.subplot(3,1,2)
    # plt.plot(np.arange(spe_width), spe_nomspike_x[i,...])
    # plt.subplot(3,1,3)
    # plt.plot(np.arange(spe_width), elp_mspike_x[2,...,0,0])
    # plt.show()
else:
    elp_nomspike_x = np.load(os.path.join(path_in, 'elpspe_v1_x.npy'))

# elp_dataset = pd.read_csv(path_nospike_elp)
# elp_nomspike_x = elp_dataset.loc[elp_dataset.loc[:,"class"]==0,[s for s in elp_dataset.columns if re.match("^spe[0-9]+$", s) is not None]]
elp_nomspike_x = elp_nomspike_x[...,None,None]

# load dataset x
elp_mspike_x = np.expand_dims(if_x[...,0], axis=(-1,-2))

# load dataset y
elp_mspike_y = np.expand_dims(if_y, axis=-2)
# invert y maps
elp_mspike_y = np.expand_dims(elp_mspike_y.max(axis=-1), axis=-1)-elp_mspike_y

# for generating random mspikes: get mean and std for spike loc and size
def getMSpikesParameters():
    ref_maps = elp_mspike_y[:,:,0,:].max(-1)
    d_distrib_curve, d_distrib = np.where(ref_maps==1)
    distrib_df = pd.DataFrame(dict(curve=d_distrib_curve, positions=d_distrib))
    center_positions = distrib_df.groupby('curve').mean()
    sizes = distrib_df.groupby('curve').count()
    maps_distrib_dict = dict(center_mean=float(center_positions.mean()),
                             center_std=float(center_positions.std()),
                             size_mean=float(sizes.mean()),
                             size_std=float(sizes.std()))
    return maps_distrib_dict
    
def generateFakeMap(maps_distrib_dict, rng = np.random):
    center = int(np.round(rng.normal(loc=maps_distrib_dict['center_mean'], scale=maps_distrib_dict['center_std'])))
    size = int(np.round(rng.normal(loc=maps_distrib_dict['size_mean'], scale=maps_distrib_dict['size_std'])))
    start_pos = center-size//2
    end_pos = start_pos+size+1
    fake_maps = np.zeros((1,spe_width,1,1))
    fake_maps[0,start_pos:end_pos,0,0] = 1
    fake_maps[0,start_pos:end_pos,0,0] = 1
    return fake_maps

# at each batch:
# X: elp without mspike
# Yx : elp with mspike
# Yy : annotations for elp with mspike

# class ELPGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, elp_nomspike_x, elp_mspike_x, elp_mspike_y, steps_per_epoch = 100, batch_size = 32, seed=42):
#         self.elp_nomspike_x      = elp_nomspike_x
#         self.elp_mspike_x        = elp_mspike_x
#         self.elp_mspike_y        = elp_mspike_y
#         self.steps_per_epoch     = steps_per_epoch
#         self.batch_size          = batch_size
#         self.seed                = seed
#         self.rng                 = np.random.RandomState(seed)
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return self.steps_per_epoch

#     def on_epoch_end(self):
#         return

#     def __getitem__(self, index):
#         # un batch de X = elp no mspike
#         # un batch de Y = elp mspike, mspike positions
#         X_indices = self.rng.choice(np.arange(self.elp_nomspike_x.shape[0]), size=self.batch_size)
#         Y_indices = self.rng.choice(np.arange(self.elp_mspike_x.shape[0]), size=self.batch_size)

#         return self.elp_nomspike_x[X_indices,...], (self.elp_mspike_x[Y_indices,...], self.elp_mspike_y[Y_indices,...].max(axis=-1)[...,None])

# generate fake dataset
# repeat dataset with smallest number of samples
elp_mspike_x = np.repeat(elp_mspike_x, repeats = np.ceil(elp_nomspike_x.shape[0] / elp_mspike_x.shape[0]), axis=0)
elp_mspike_y = np.repeat(elp_mspike_y, repeats = np.ceil(elp_nomspike_x.shape[0] / elp_mspike_y.shape[0]), axis=0)
elp_mspike_x = elp_mspike_x[:elp_nomspike_x.shape[0],...]
elp_mspike_y = elp_mspike_y[:elp_nomspike_x.shape[0],...]
# get only ref curve for mspikes annotations
elp_mspike_y = np.expand_dims(elp_mspike_y.max(axis=-1), axis=-1)

if DEBUG:
    elp_nomspike_x = elp_nomspike_x[:FLAGS.batch_size*4,...]
    elp_mspike_x = elp_mspike_x[:FLAGS.batch_size*4,...]
    elp_mspike_y = elp_mspike_y[:FLAGS.batch_size*4,...]

dataset = tf.data.Dataset.from_tensor_slices((elp_nomspike_x.astype(np.float32), elp_mspike_x.astype(np.float32), elp_mspike_y.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=1024).batch(FLAGS.batch_size)

# reshape data for this predictive task
# only keep first dimension

# data_generator = ELPGenerator(elp_nomspike_x.astype(np.float32), elp_mspike_x.astype(np.float32), elp_mspike_y.astype(np.float32),
#                               steps_per_epoch = FLAGS.steps_per_epoch, batch_size = FLAGS.batch_size)    

if False:
    for data in dataset.as_numpy_iterator():
        X, Yx, Yy = data
        break
    
    i=3
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.arange(spe_width), X[i,...,0,0])
    plt.subplot(3,1,2)
    plt.plot(np.arange(spe_width), Yx[i,...,0,0])
    plt.subplot(3,1,3)
    plt.plot(np.arange(spe_width), Yy[i,...,0,0])
    plt.show()

# %%

# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return tf.keras.backend.mean(y_true * y_pred)

# on va créer un GAN u-net
# il prend 2 inputs: latent_dim (=8 par défaut) et maps (dim 304x1x5)
# latent_dim -> (dense) -> 304 valeurs -> reshape en 304x1x1
# puis concat 304 valeurs & maps -> 304x1x6
# puis u-net -> elp de 304x1x1 valeurs
# puis discriminateur : prend en input 304x1x1 et maps et doit dire si ok

# creates a 1d conv block, with conv -> batch norm -> relu activation
def conv1d_block(input_tensor, n_filters, layers, name, kernel_size=3, batchnorm=True):
    x = input_tensor
    for l in range (layers):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last', name = name + "conv{}".format(l+1)) (x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(name = name + "batchnorm{}".format(l+1)) (x)
        x = tf.keras.layers.Activation("relu", name = name + "dropout{}".format(l+1)) (x)
    return x

def get_encoder(x, n_filters=32, blocks = 4, layers_per_block = 2, dropout=0.05, batchnorm=True):
    for b in range(blocks):
        x = conv1d_block(input_tensor = x, n_filters = int(n_filters*np.power(2,b)), layers = layers_per_block, kernel_size = 3, batchnorm = batchnorm, name = "encoder_block{}_".format(b+1))
        x = tf.keras.layers.MaxPooling2D((2,1)) (x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout) (x)
            
    return x

def get_decoder(x, n_filters=32, blocks = 4, layers_per_block = 2, batchnorm=True):
    for b in range(blocks):
        x = tf.keras.layers.Conv2DTranspose(int(n_filters*np.power(2,blocks-(b+1))), (3,1), strides=(2,1), padding='same') (x)
        x = conv1d_block(input_tensor = x, n_filters = int(n_filters*np.power(2,blocks-(b+1))), layers = layers_per_block, kernel_size = 3, batchnorm = batchnorm, name = "decoder_block{}_".format(b+1))
        
    return x

def get_unet_structure(x, blocks=4, n_filters=32, kernel_size=3, layers_per_block=2, dropout=0.05, batchnorm=True):
    # contracting path
    conv_blocks = []
    
    # encoder
    last_pb = x
    for b in range(blocks+1):
        dropout_mult=1.
        if b==0:
            dropout_mult=.5
        cb = conv1d_block(last_pb, n_filters=n_filters*np.power(2,b), kernel_size=kernel_size, layers=layers_per_block, batchnorm=batchnorm, name="enc_block{}_".format(b+1))
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
        cb = conv1d_block(ub, n_filters=n_filters*np.power(2,blocks-b-1), kernel_size=kernel_size, layers=layers_per_block, batchnorm=batchnorm, name="dec_block{}_".format(b+1))
        conv_blocks.append(cb)
    
    return conv_blocks[-1]

def define_generator(n_filters = 32, blocks = 4, layers_per_block = 2, name = "generator"):
    # elp_input = tf.keras.layers.Input((spe_width,1,1))
    # map_input = tf.keras.layers.Input((spe_width,1,1))
    # x = tf.keras.layers.Concatenate(axis=-1) ([elp_input, map_input])
    
    concat_input = tf.keras.layers.Input((spe_width,1,2))
    x = concat_input
    
    # run through u-net
    x = get_unet_structure(x, blocks=blocks, n_filters=n_filters, layers_per_block=layers_per_block)
    
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid') (x)
    
    generator = tf.keras.models.Model(concat_input, outputs, name = name)
    # generator = tf.keras.models.Model([elp_input, map_input], outputs, name = "generator")

    return generator

def define_discriminator(n_filters = 32, blocks = 4, layers_per_block = 2, name = "discriminator"):
    # elp_input = tf.keras.layers.Input((spe_width,1,1))
    # map_input = tf.keras.layers.Input((spe_width,1,1))
    
    # x = tf.keras.layers.Concatenate(axis=-1) ([elp_input, map_input])

    concat_input = tf.keras.layers.Input((spe_width,1,2))
    x = concat_input
    
    x = get_encoder(x, n_filters=n_filters,blocks=blocks,layers_per_block=layers_per_block)
    
    # x = tf.keras.layers.Flatten() (x)
    x = tf.keras.layers.GlobalMaxPooling2D() (x)
    
    if FLAGS.loss == 'wasserstein':
        last_layer_activation = 'tanh'
        print("Initializing {} with 'tanh' activation".format(name))
    else:
        last_layer_activation = 'sigmoid'
        print("Initializing {} with 'sigmoid' activation".format(name))
    
    outputs = tf.keras.layers.Dense(1, activation=last_layer_activation) (x)
    
    # discriminator = tf.keras.models.Model([elp_input, map_input], outputs, name = "discriminator")
    discriminator = tf.keras.models.Model(concat_input, outputs, name = name)

    return discriminator

class GAN(tf.keras.Model):
    def __init__(self, discriminator_x, discriminator_y, generator_xy, generator_yx, LAMBDA = 10):
        super(GAN, self).__init__()
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self.generator_xy = generator_xy
        self.generator_yx = generator_yx
        self.LAMBDA = LAMBDA

    def compile(self, dx_optimizer, dy_optimizer, gxy_optimizer, gyx_optimizer, loss_fn):
        super(GAN, self).compile()
        self.dx_optimizer = dx_optimizer
        self.dy_optimizer = dy_optimizer
        self.gxy_optimizer = gxy_optimizer
        self.gyx_optimizer = gyx_optimizer
        self.loss_fn = loss_fn
        self.dx_loss_metric = tf.keras.metrics.Mean(name="dx_loss")
        self.dy_loss_metric = tf.keras.metrics.Mean(name="dy_loss")
        self.gxy_loss_metric = tf.keras.metrics.Mean(name="gxy_loss")
        self.gyx_loss_metric = tf.keras.metrics.Mean(name="gyx_loss")

    @property
    def metrics(self):
        return [self.dx_loss_metric, self.dy_loss_metric, self.gxy_loss_metric, self.gyx_loss_metric]
    
    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1
    
    def generator_loss(self, generated):
        return self.loss_fn(tf.ones_like(generated), generated)
    
    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss
    
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_fn(tf.ones_like(real), real)

        generated_loss = self.loss_fn(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def train_step(self, data):
        # data = data_generator.__getitem__(0)
            
        # Unpack the data.
        real_x, real_y, y_annot = data
        
        # print("real_x shape: {}".format(real_x.shape))
        # print("real_y shape: {}".format(real_y.shape))
        # print("y_annot shape: {}".format(y_annot.shape))
        
        # run batch training
        with tf.GradientTape(persistent=True) as tape:
            # Generator G (XY) translates X -> Y
            # Generator F (YX) translates Y -> X.
            
            # generate fake elp' from real elp, and reverse to initial elp (using real elp annots)
            fake_y = self.generator_xy(tf.concat([real_x, y_annot], axis=-1), training=True)
            # print("fake_y shape: {}".format(fake_y.shape))
            cycled_x = self.generator_yx(tf.concat([fake_y, y_annot], axis=-1), training=True)
            # print("cycled_x shape: {}".format(cycled_x.shape))
          
            # generate fake elp from real elp', and reverse to initial elp' (using real elp' annots)
            fake_x = self.generator_yx(tf.concat([real_y, y_annot], axis=-1), training=True)
            # print("fake_x shape: {}".format(fake_x.shape))
            cycled_y = self.generator_xy(tf.concat([fake_x, y_annot], axis=-1), training=True)
            # print("cycled_y shape: {}".format(cycled_y.shape))
          
            # same_x and same_y are used for identity loss.
            same_x = self.generator_yx(tf.concat([real_x, y_annot], axis=-1), training=True)
            same_y = self.generator_xy(tf.concat([real_y, y_annot], axis=-1), training=True)
          
            disc_real_x = self.discriminator_x(tf.concat([real_x, y_annot], axis=-1), training=True)
            disc_real_y = self.discriminator_y(tf.concat([real_y, y_annot], axis=-1), training=True)
          
            disc_fake_x = self.discriminator_x(tf.concat([fake_x, y_annot], axis=-1), training=True)
            disc_fake_y = self.discriminator_y(tf.concat([fake_y, y_annot], axis=-1), training=True)
          
            # calculate the loss
            gen_xy_loss = self.generator_loss(disc_fake_y)
            gen_yx_loss = self.generator_loss(disc_fake_x)
            
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
          
            # Total generator loss = adversarial loss + cycle loss
            total_gen_xy_loss = gen_xy_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_yx_loss = gen_yx_loss + total_cycle_loss + self.identity_loss(real_x, same_x)
            
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
            
        # Calculate the gradients for generator and discriminator
        generator_xy_gradients = tape.gradient(total_gen_xy_loss, 
                                               self.generator_xy.trainable_variables)
        generator_yx_gradients = tape.gradient(total_gen_yx_loss, 
                                               self.generator_yx.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                  self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.gxy_optimizer.apply_gradients(zip(generator_xy_gradients, 
                                               self.generator_xy.trainable_variables))
        
        self.gyx_optimizer.apply_gradients(zip(generator_yx_gradients, 
                                               self.generator_yx.trainable_variables))
        
        self.dx_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                              self.discriminator_x.trainable_variables))
        
        self.dy_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                              self.discriminator_y.trainable_variables))
            
        # Update metrics
        self.dx_loss_metric.update_state(disc_x_loss)
        self.dy_loss_metric.update_state(disc_y_loss)
        self.gxy_loss_metric.update_state(total_gen_xy_loss)
        self.gyx_loss_metric.update_state(total_gen_yx_loss)
        return {
            "dx_loss": self.dx_loss_metric.result(),
            "dy_loss": self.dy_loss_metric.result(),
            "gxy_loss": self.gxy_loss_metric.result(),
            "gyx_loss": self.gyx_loss_metric.result(),
        }
    
class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, path, X_dataset, maps_distrib_dict, every = 1, num_img=3):
        self.path = path
        self.X_dataset = X_dataset
        self.maps_distrib_dict = maps_distrib_dict
        self.every = every
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every > 0:
            return
        # get random elp
        random_elp = self.X_dataset[[np.random.choice(np.arange(self.X_dataset.shape[0]))],...]
        # get random map
        random_maps = np.concatenate([generateFakeMap(self.maps_distrib_dict) for i in range(self.num_img)])
        # random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        generated_images = self.model.generator_xy.predict(tf.concat([random_elp, random_maps], axis=-1))[...,0,0]
        # save figures
        for i in range(self.num_img):
            plt.figure(figsize=(6,6))
            plt.subplot(2,1,1)
            plt.plot(np.arange(random_elp.shape[1]), random_elp[i,...,0,0])
            plt.ylim(0,1)
            plt.subplot(2,1,2)
            plt.fill_between(np.where(random_maps[i,...]>0)[0],
                             np.repeat(1, np.sum(random_maps[i,...]>0)),
                             color='red')
            plt.plot(np.arange(generated_images.shape[1]), generated_images[i,...])
            plt.ylim(np.min([0, np.min(generated_images[i,...])]),
                     np.max([1, np.max(generated_images[i,...])]))
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1)))
            plt.close()
    
# %%

# define HP
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
# STEPS_PER_EPOCH = FLAGS.steps_per_epoch
MODEL_NAME = "cyclegan_additive_elp_v1"

# create folders
model_path = os.path.join(path_out,MODEL_NAME,"model")
imgs_path = os.path.join(path_out,MODEL_NAME,"imgs")
os.makedirs(imgs_path, exist_ok=True)

# instanciate models
discriminator_X = define_discriminator(n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block, name="discriminator_x")
discriminator_Y = define_discriminator(n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block, name="discriminator_y")
generator_XtoY = define_generator(n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block, name="generator_xy")
generator_YtoX = define_generator(n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block, name="generator_yx")

# discriminator_X.summary()
# generator_YtoX.summary()

# tf.keras.utils.plot_model(model=mdl, to_file=re.sub("\\\\","/",model_path+"_generator.jpg"), show_shapes=True, show_layer_names=True, dpi=90)

# if False:
#     random_latent_vectors = np.random.normal(size=(1, LATENT_DIM))
#     generated_images = generator.predict(random_latent_vectors)
#     generated_images = postProc(generated_images)

#     i=0
#     plt.figure(figsize=(6,4))
#     plt.plot(np.arange(generated_images.shape[1]), generated_images[i,...])
#     plt.ylim((0,1))
#     plt.tight_layout()
#     plt.show()
    
#     generated_images = dataset[[1],...]
#     generated_images = postProc(generated_images)
    
#     i=0
#     plt.figure(figsize=(6,4))
#     plt.plot(np.arange(dataset.shape[1]), generated_images[i,...])
#     plt.ylim((0,1))
#     plt.tight_layout()
#     plt.show()
    
gan = GAN(discriminator_x=discriminator_X,
          discriminator_y=discriminator_Y,
          generator_xy=generator_XtoY,
          generator_yx=generator_YtoX,
          LAMBDA=10)

if FLAGS.loss == "wasserstein":
    loss_fn = wasserstein_loss
elif FLAGS.loss == "basic":
    loss_fn = tf.keras.losses.BinaryCrossentropy()

gan.compile(
    dx_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=.5),
    dy_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=.5),
    gxy_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=.5),
    gyx_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=.5),
    loss_fn=loss_fn,
)

callbacks = [
    GANMonitor(path=imgs_path, X_dataset=elp_nomspike_x, maps_distrib_dict=getMSpikesParameters(), every = 1, num_img=1),
    tf.keras.callbacks.ModelCheckpoint(model_path+".h5", verbose=1, save_weights_only=False)
]

results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)

with open(model_path+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%

# X, (Yx, Yy) = data_generator.__getitem__(0)

# gan.generator_xy.predict(tf.concat([X, Yy], axis=-1))
# gan.generator_xy(tf.concat([X, Yy], axis=-1))




































