# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:04:31 2021

@author: admin
"""

import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
plt.ioff()

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--step", type=str, default='test')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--latent_dim", type=int, default=8)

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

if True:
    path_in = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    path_out = r'C:\Users\admin\Documents\Capillarys\data\delta_gan\out'
# else:
#     path_in = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/data'
#     path_out = '/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/out'
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(path_in,'if_v1_x.npy'))
# if_y = np.load(os.path.join(path_in,'if_v1_y.npy'))

spe_width = 304

# load dataset
if FLAGS.curve_scale == "standardized":
    dataset = if_x[...,0]
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    data_std[data_std==0]=1
    dataset = (dataset-data_mean)/data_std
    dataset = np.expand_dims(dataset, axis=(-1,-2))
    
    def postProc(spe_curve):
        if len(spe_curve.shape) != 2:
            spe_curve = spe_curve.reshape((spe_curve.shape[0], spe_width))
        return spe_curve, spe_curve * data_std + data_mean
else:
    dataset = np.expand_dims(if_x[...,0], axis=(-1,-2))
    
    def postProc(spe_curve):
        if len(spe_curve.shape) != 2:
            spe_curve = spe_curve.reshape((spe_curve.shape[0], spe_width))
        return spe_curve, spe_curve


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

def define_generator(latent_dim, n_filters = 32, blocks = 4, layers_per_block = 2):
    latent_input = tf.keras.layers.Input(latent_dim)
    x = latent_input
    
    maxpooling_factor = np.power(2, blocks)
    reduced_dim = spe_width // maxpooling_factor
    reduced_n_filters = n_filters * maxpooling_factor
    firstconv_dim = (reduced_dim, 1, reduced_n_filters)
    
    x = tf.keras.layers.Dense(np.prod(firstconv_dim), kernel_initializer='he_normal') (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Reshape(firstconv_dim) (x)
    
    x = get_decoder(x, n_filters=n_filters,blocks=blocks,layers_per_block=layers_per_block)
    
    if FLAGS.curve_scale == 'standardized':
        last_layer_activation = None
        print("Initializing generator with 'none' activation")
    else:
        last_layer_activation = "sigmoid"
        print("Initializing generator with 'sigmoid' activation")
        
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation=last_layer_activation) (x)
    
    generator = tf.keras.models.Model(latent_input, outputs, name = "generator")

    return generator

def define_discriminator(n_filters = 32, blocks = 4, layers_per_block = 2):
    input_sample = tf.keras.layers.Input((spe_width,1,1))
    x = input_sample
    
    x = get_encoder(x, n_filters=n_filters,blocks=blocks,layers_per_block=layers_per_block)
    
    x = tf.keras.layers.Flatten() (x)
    
    if FLAGS.loss == 'wasserstein':
        last_layer_activation = 'tanh'
        print("Initializing discriminator with 'tanh' activation")
    else:
        last_layer_activation = 'sigmoid'
        print("Initializing discriminator with 'sigmoid' activation")
    
    outputs = tf.keras.layers.Dense(1, activation=last_layer_activation) (x)
    
    discriminator = tf.keras.models.Model(input_sample, outputs, name = "discriminator")

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

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
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
            predictions = self.discriminator(self.generator(random_latent_vectors))
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

    def on_epoch_end(self, epoch, logs=None):
        # generate fake samples
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        # random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        generated_images = self.model.generator.predict(random_latent_vectors)
        generated_images_raw, generated_images_processed = postProc(generated_images) # re-scale curves
        # save figures
        plt.figure(figsize=(12,4*self.num_img))
        for i in range(self.num_img):
            plt.subplot(self.num_img,2,i*2+1)
            plt.plot(np.arange(generated_images_raw.shape[1]), generated_images_raw[i,...])
            plt.subplot(self.num_img,2,i*2+2)
            plt.plot(np.arange(generated_images_processed.shape[1]), generated_images_processed[i,...])
            plt.ylim((0,1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1)))
        plt.close()
    
# %%

# define HP
LATENT_DIM = FLAGS.latent_dim
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
MODEL_NAME = "model_test.h5"

# instanciate models
discriminator = define_discriminator(n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)
generator = define_generator(latent_dim = LATENT_DIM, n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)

discriminator.summary()
generator.summary()

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
    GANMonitor(os.path.join(path_out,"demos"), num_img=3, latent_dim=LATENT_DIM),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,"models",MODEL_NAME), verbose=1, save_weights_only=False)
]

gan.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    
# %%
