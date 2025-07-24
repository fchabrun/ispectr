# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:04:31 2021

@author: admin
"""

# architecture globale pensée :
# 1. on génère une ELP grâce à un premier GAN
#    pour la générer, on lui donne
#    - du bruit aléatoire
#    - la localisation du pic
#    - le type de pic (G, A, M, k, l)
# 2. on utilise un 2e GAN, cycle GAN qui va générer les courbes G, A, M, k, l depuis
#    la courbe initiale générée par le 1er GAN
#    la localisation et le type de pic
#    la courbe que l'on veut (G, A, M, k, l) (ou alors on fait 5 cycleGANs)

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
plt.ioff()

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--debug", type=int, default=0)
# parser.add_argument("--step", type=str, default='test')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--latent_dim", type=int, default=8)
parser.add_argument("--generator_architecture", type=str, default="u-net")

parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--n_filters", type=int, default=32)
parser.add_argument("--n_layers_per_block", type=int, default=1)

parser.add_argument("--learning_rate", type=float, default=1e-04)
parser.add_argument("--curve_scale", type=str, default="1-normalized", help='scale for curve values, either "1-normalized" or "standardized"')
parser.add_argument("--loss", type=str, default="basic", help='loss used, either "basic" or "wasserstein"')

FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if FLAGS.host=="local":
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = r'C:\Users\admin\Documents\Capillarys\data\delta_gan\out'
else:
    path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/data'
    path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/out'
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

spe_width = 304

# load dataset x
if FLAGS.curve_scale == "standardized":
    dataset_x = if_x[...,0]
    data_mean = dataset_x.mean(axis=0)
    data_std = dataset_x.std(axis=0)
    data_std[data_std==0]=1
    dataset_x = (dataset_x-data_mean)/data_std
    dataset_x = np.expand_dims(dataset_x, axis=(-1,-2))
    
    def postProc(spe_curve):
        if len(spe_curve.shape) != 2:
            spe_curve = spe_curve.reshape((spe_curve.shape[0], spe_width))
        return spe_curve, spe_curve * data_std + data_mean
else:
    dataset_x = np.expand_dims(if_x[...,0], axis=(-1,-2))
    
    def postProc(spe_curve):
        if len(spe_curve.shape) != 2:
            spe_curve = spe_curve.reshape((spe_curve.shape[0], spe_width))
        return spe_curve, spe_curve

# load dataset y
dataset_y = np.expand_dims(if_y, axis=-2)
# invert y maps
dataset_y = np.expand_dims(dataset_y.max(axis=-1), axis=-1)-dataset_y
    
# determine histograms of distribution of different types of Ig
# for generating realistic m-spikes
# based on the heavy chain type
# because we suspect that the light chain type has no major impact? (except when free light chains, but we won't generate those)
maps_distrib_dict = dict()
for d in range(0,3):
    d_distrib_curve, d_distrib = np.where(dataset_y[:,:,0,d]==1)
    distrib_df = pd.DataFrame(dict(curve=d_distrib_curve, positions=d_distrib))
    center_positions = distrib_df.groupby('curve').mean()
    sizes = distrib_df.groupby('curve').count()
    maps_distrib_dict[d] = dict(center_mean=float(center_positions.mean()),
                                center_std=float(center_positions.std()),
                                size_mean=float(sizes.mean()),
                                size_std=float(sizes.std()))
    
# now we can make our function to generate a random, realistic map
def generateFakeMap(heavy, light, rng):
    center = int(np.round(rng.normal(loc=maps_distrib_dict[heavy]['center_mean'], scale=maps_distrib_dict[heavy]['center_std'])))
    size = int(np.round(rng.normal(loc=maps_distrib_dict[heavy]['size_mean'], scale=maps_distrib_dict[heavy]['size_std'])))
    start_pos = center-size//2
    end_pos = start_pos+size+1
    fake_maps = np.zeros((1,spe_width,1,5))
    fake_maps[0,start_pos:end_pos,:,heavy] = 1
    fake_maps[0,start_pos:end_pos,:,light] = 1
    return fake_maps

# # we must define a function to generate random food for the generator
# def getFakeBatch(latent_dim, batch_size):
#     input_noise = np.random.normal(size=(batch_size,latent_dim))
#     input_maps = np.concatenate([generateFakeMap(int(heavy),int(light),np.random) for heavy,light in zip(np.floor(np.random.uniform(low=0, high=3, size=batch_size)),np.floor(np.random.uniform(low=3, high=5, size=batch_size)))])
#     return input_noise, input_maps

# create a generator for 

if False:
    i = 0
    if False:
        maps = dataset_y[[i],...]
    # maps = generateFakeMap(0, 3, np.random)
    plt.figure()
    plt.subplot(6,1,1)
    plt.plot(np.arange(spe_width), dataset_x[i,...,0,0])
    plt.ylim(0,1)
    for j in range(5):
        plt.subplot(6,1,j+2)
        plt.plot(np.arange(spe_width), maps[i,...,0,j])
        plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

dataset = tf.data.Dataset.from_tensor_slices((dataset_x.astype(np.float32), dataset_y.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=256).batch(FLAGS.batch_size)

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

def define_generator(latent_dim, maps_depth, generator_architecture, n_filters = 32, blocks = 4, layers_per_block = 2):
    latent_input = tf.keras.layers.Input(latent_dim)
    maps_input = tf.keras.layers.Input((spe_width,1,maps_depth))
    
    if generator_architecture == "u-net":
        print("Using u-net generator architecture")
        x = tf.keras.layers.Dense(spe_width, kernel_initializer='he_normal') (latent_input)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Reshape((spe_width,1,1)) (x)
        
        x = tf.keras.layers.Concatenate(axis=-1) ([x, maps_input])
        
        # run through u-net
        x = get_unet_structure(x, blocks=blocks, n_filters=n_filters, layers_per_block=layers_per_block)
    else:
        print("Using regular generator architecture")
        # reshape maps input using maxpooling
        x = tf.keras.layers.MaxPooling2D((np.power(2,blocks),1)) (maps_input)
        x = tf.keras.layers.Flatten() (x)
        x = tf.keras.layers.Concatenate() ([latent_input, x])
        
        maxpooling_factor = np.power(2, blocks)
        reduced_dim = spe_width // maxpooling_factor
        reduced_n_filters = n_filters * maxpooling_factor
        firstconv_dim = (reduced_dim, 1, reduced_n_filters)
    
        x = tf.keras.layers.Dense(np.prod(firstconv_dim), kernel_initializer='he_normal') (x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Reshape(firstconv_dim) (x)
        
        # run through u-net
        x = get_decoder(x, blocks=blocks, n_filters=n_filters, layers_per_block=layers_per_block)
    
    if FLAGS.curve_scale == 'standardized':
        last_layer_activation = None
        print("Initializing generator with 'none' activation")
    else:
        last_layer_activation = "sigmoid"
        print("Initializing generator with 'sigmoid' activation")
        
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation=last_layer_activation) (x)
    
    generator = tf.keras.models.Model([latent_input, maps_input], outputs, name = "generator")

    return generator

def define_discriminator(maps_depth, n_filters = 32, blocks = 4, layers_per_block = 2):
    sample_input = tf.keras.layers.Input((spe_width,1,1))
    maps_input = tf.keras.layers.Input((spe_width,1,maps_depth))
    x = tf.keras.layers.Concatenate(axis=-1) ([sample_input, maps_input])
    
    x = get_encoder(x, n_filters=n_filters,blocks=blocks,layers_per_block=layers_per_block)
    
    # x = tf.keras.layers.Flatten() (x)
    x = tf.keras.layers.GlobalMaxPooling2D() (x)
    
    if FLAGS.loss == 'wasserstein':
        last_layer_activation = 'tanh'
        print("Initializing discriminator with 'tanh' activation")
    else:
        last_layer_activation = 'sigmoid'
        print("Initializing discriminator with 'sigmoid' activation")
    
    outputs = tf.keras.layers.Dense(1, activation=last_layer_activation) (x)
    
    discriminator = tf.keras.models.Model([sample_input, maps_input], outputs, name = "discriminator")

    return discriminator

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        # for data in dataset.as_numpy_iterator():
        #     break
            
        # Unpack the data.
        batch_samples, batch_maps = data
        # and get batch size
        batch_size = tf.shape(batch_samples)[0]
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake samples using real maps
        generated_samples = self.generator([random_latent_vectors, batch_maps])

        # Combine them with real samples
        combined_samples = tf.concat([generated_samples, batch_samples], axis=0)
        combined_maps = tf.concat([batch_maps, batch_maps], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_samples, combined_maps])
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))
        
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator([self.generator([random_latent_vectors, batch_maps]), batch_maps])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
    
class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, path, num_img=3, latent_dim=128):
        self.path = path
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, every = 10, logs=None):
        if epoch % every > 0:
            return
        # generate fake samples
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        # with maps
        random_maps = np.concatenate([generateFakeMap(int(heavy),int(light),np.random) for heavy,light in zip(np.floor(np.random.uniform(low=0, high=3, size=self.num_img)),np.floor(np.random.uniform(low=3, high=5, size=self.num_img)))])
        # random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        generated_images = self.model.generator.predict([random_latent_vectors, random_maps])[...,0,0]
        # save figures
        ref_maps = random_maps[:,:,0,:].max(axis=-1)
        for i in range(self.num_img):
            plt.figure(figsize=(6,4))
            plt.fill_between(np.where(ref_maps[i,...]>0)[0],
                             np.repeat(1, np.sum(ref_maps[i,...]>0)),
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
LATENT_DIM = FLAGS.latent_dim
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
MODEL_NAME = "cgan_unet_elp_v1"

# create folders
model_path = os.path.join(path_out,MODEL_NAME,"model")
imgs_path = os.path.join(path_out,MODEL_NAME,"imgs")
os.makedirs(imgs_path, exist_ok=True)

# instanciate models
discriminator = define_discriminator(maps_depth = 5, n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)
generator = define_generator(latent_dim = LATENT_DIM, maps_depth = 5, generator_architecture = FLAGS.generator_architecture, n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)

discriminator.summary()
generator.summary()

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
    
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

if FLAGS.loss == "wasserstein":
    loss_fn = wasserstein_loss
elif FLAGS.loss == "basic":
    loss_fn = tf.keras.losses.BinaryCrossentropy()

gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
    loss_fn=loss_fn,
)

callbacks = [
    GANMonitor(imgs_path, num_img=1, latent_dim=LATENT_DIM),
    tf.keras.callbacks.ModelCheckpoint(model_path+".h5", verbose=1, save_weights_only=False)
]

results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)

with open(model_path+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%
