# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:58:26 2021

@author: admin
"""

# un gan pour créer les 6 courbes + annotations
# generator : input  = random noise (LATENT_DIM,)
#             output = (304,2,6)

import argparse
import numpy as np
# import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
plt.ioff()

import tensorflow as tf
# import tensorflow_addons as tfa

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--debug", type=int, default=0)
# parser.add_argument("--step", type=str, default='test')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--generator_architecture", type=str, default="conv")

parser.add_argument("--generate_what", type=str, default="elp_alone")

parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--n_filters", type=int, default=32)
parser.add_argument("--n_layers_per_block", type=int, default=1)

parser.add_argument("--learning_rate", type=float, default=1e-04)
# parser.add_argument("--loss", type=str, default="regular", help='loss used, either "regular" or "wasserstein"')
parser.add_argument("--loss", type=str, default="wasserstein", help='loss used, either "regular" or "wasserstein"')

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
if_x = np.expand_dims(if_x, axis=-2)

# load dataset y
if_y = np.expand_dims(if_y, axis=-2)
# invert y maps
if_y = np.expand_dims(if_y.max(axis=-1), axis=-1)-if_y
# add ref curve map
if_y = np.concatenate([np.expand_dims(if_y.max(axis=-1), axis=-1), if_y], axis=-1)

# concatenate altogether
if_all = np.concatenate([if_x, if_y], axis=-2)

if FLAGS.generate_what == "elp_alone":
    dataset = tf.data.Dataset.from_tensor_slices(if_x[...,[0]].astype(np.float32))
elif FLAGS.generate_what == "all+annotations":
    dataset = tf.data.Dataset.from_tensor_slices(if_all.astype(np.float32))
dataset = dataset.shuffle(buffer_size=256).batch(FLAGS.batch_size)

# function for plotting a sample (used for the monitoring callback)
def plotSample(sample_data, save = None):
    from matplotlib import pyplot as plt
    if sample_data.shape == (304,2,6):
        plot_colors = ('black','red',)
        plt.figure(figsize=(12,8))
        for j in range(6):
            crv_x = np.arange(sample_data.shape[0])
            crv_v = sample_data[...,0,j]
            crv_a = (sample_data[...,1,j]>=.5)*1
            plt.subplot(6,2,j*2+1)
            # fill area
            plt.fill_between(crv_x[crv_a==1], crv_v[crv_a==1], color='#ffaaaa')
            # plot curves
            start_loc = 0
            start_a = crv_a[start_loc]
            cur_loc = 1
            while True:
                if cur_loc >= crv_x.shape[0]:
                    # print("j={}; plotting from {} to {} with color {} (end)".format(j,start_a,cur_loc,plot_colors[start_a]))
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                    break
                new_a = crv_a[cur_loc]
                if new_a != start_a:
                    # print("j={}; plotting from {} to {} with color {}".format(j,start_a,cur_loc,plot_colors[start_a]))
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                    start_loc = cur_loc
                    start_a = new_a
                cur_loc += 1
            plt.subplot(6,2,j*2+2)
            plt.plot(crv_x, sample_data[...,1,j], color="#333333")
    elif sample_data.shape == (304,1,1):
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(sample_data.shape[0]), sample_data[...,0,0])
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

# test
if False:
    for batch in dataset.as_numpy_iterator():
        break
    sample_data = batch[0,...]
    plotSample(sample_data)
    
        

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

# @tf.keras.utils.register_keras_serializable()
# class AddPositionEmbs(tf.keras.layers.Layer):
#     """Adds (optionally learned) positional embeddings to the inputs."""

#     def build(self, input_shape):
#         assert (
#             len(input_shape) == 3
#         ), f"Number of dimensions should be 3, got {len(input_shape)}"
#         self.pe = tf.Variable(
#             name="pos_embedding",
#             initial_value=tf.random_normal_initializer(stddev=0.06)(
#                 shape=(1, input_shape[1], input_shape[2])
#             ),
#             dtype="float32",
#             trainable=True,
#         )

#     def call(self, inputs):
#         return inputs + tf.cast(self.pe, dtype=inputs.dtype)

#     def get_config(self):
#         config = super().get_config()
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
# @tf.keras.utils.register_keras_serializable()
# class MultiHeadSelfAttention(tf.keras.layers.Layer):
#     def __init__(self, *args, num_heads, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_heads = num_heads

#     def build(self, input_shape):
#         hidden_size = input_shape[-1]
#         num_heads = self.num_heads
#         if hidden_size % num_heads != 0:
#             raise ValueError(
#                 f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
#             )
#         self.hidden_size = hidden_size
#         self.projection_dim = hidden_size // num_heads
#         self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
#         self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
#         self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
#         self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

#     # pylint: disable=no-self-use
#     def attention(self, query, key, value):
#         score = tf.matmul(query, key, transpose_b=True)
#         dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
#         scaled_score = score / tf.math.sqrt(dim_key)
#         weights = tf.nn.softmax(scaled_score, axis=-1)
#         output = tf.matmul(weights, value)
#         return output, weights

#     def separate_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#         query = self.query_dense(inputs)
#         key = self.key_dense(inputs)
#         value = self.value_dense(inputs)
#         query = self.separate_heads(query, batch_size)
#         key = self.separate_heads(key, batch_size)
#         value = self.separate_heads(value, batch_size)

#         attention, weights = self.attention(query, key, value)
#         attention = tf.transpose(attention, perm=[0, 2, 1, 3])
#         concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
#         output = self.combine_heads(concat_attention)
#         return output, weights

#     def get_config(self):
#         config = super().get_config()
#         config.update({"num_heads": self.num_heads})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
# @tf.keras.utils.register_keras_serializable()
# class TransformerBlock(tf.keras.layers.Layer):
#     """Implements a Transformer block."""

#     def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_heads = num_heads
#         self.mlp_dim = mlp_dim
#         self.dropout = dropout

#     def build(self, input_shape):
#         self.att = MultiHeadSelfAttention(
#             num_heads=self.num_heads,
#             name="MultiHeadDotProductAttention_1",
#         )
#         self.mlpblock = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Dense(
#                     self.mlp_dim,
#                     activation="linear",
#                     name=f"{self.name}/Dense_0",
#                 ),
#                 tf.keras.layers.Lambda(
#                     lambda x: tf.keras.activations.gelu(x, approximate=False)
#                 )
#                 if hasattr(tf.keras.activations, "gelu")
#                 else tf.keras.layers.Lambda(
#                     lambda x: tfa.activations.gelu(x, approximate=False)
#                 ),
#                 tf.keras.layers.Dropout(self.dropout),
#                 tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
#                 tf.keras.layers.Dropout(self.dropout),
#             ],
#             name="MlpBlock_3",
#         )
#         self.layernorm1 = tf.keras.layers.LayerNormalization(
#             epsilon=1e-6, name="LayerNorm_0"
#         )
#         self.layernorm2 = tf.keras.layers.LayerNormalization(
#             epsilon=1e-6, name="LayerNorm_2"
#         )
#         self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

#     def call(self, inputs, training):
#         x = self.layernorm1(inputs)
#         x, weights = self.att(x)
#         x = self.dropout_layer(x, training=training)
#         x = x + inputs
#         y = self.layernorm2(x)
#         y = self.mlpblock(y)
#         return x + y, weights

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "num_heads": self.num_heads,
#                 "mlp_dim": self.mlp_dim,
#                 "dropout": self.dropout,
#             }
#         )
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# def define_discriminator_transformer(input_size = spe_width, hidden_size = 4, patch_size = 4, num_layers = 3, num_heads = 12, mlp_dim = 16, dropout=.1, representation_size = None, include_top=True):
#     assert input_size % patch_size == 0, "input_size must be a multiple of patch_size"
#     # 304x2x6 -> 6 inputs 304x1x2
#     inputs = tf.keras.layers.Input(shape=(304, 2, 6))
#     x = inputs
#     xs = [tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1])) (x),
#           tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 1], [-1, -1, -1, 1])) (x),
#           tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 2], [-1, -1, -1, 1])) (x),
#           tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 3], [-1, -1, -1, 1])) (x),
#           tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 4], [-1, -1, -1, 1])) (x),
#           tf.keras.layers.Lambda( lambda x: tf.slice(x, [0, 0, 0, 5], [-1, -1, -1, 1])) (x)]
    
#     # embedding conv
#     xs = [tf.keras.layers.Conv2D(filters=hidden_size,kernel_size=(patch_size,1),strides=(patch_size,1),padding="valid",name="embedding{}".format(i+1)) (x) for i,x in enumerate(xs)]
#     xs = [tf.keras.layers.Reshape((x.shape[1] * x.shape[2], hidden_size)) (x) for i,x in enumerate(xs)]
#     xs = [AddPositionEmbs(name="Transformer/posembed_input{}".format(i+1)) (x) for i,x in enumerate(xs)]
#     x = tf.keras.layers.Concatenate() (xs)
#     for n in range(num_layers):
#         x, _ = TransformerBlock(
#             num_heads=num_heads,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             name=f"Transformer/encoderblock_{n}",
#             ) (x)
#     x = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6, name="Transformer/encoder_norm"
#     ) (x)
#     x = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken") (x)
#     if representation_size is not None:
#         x = tf.keras.layers.Dense(
#             representation_size, name="pre_logits", activation="tanh"
#         ) (x)
#     if include_top:
#         x = tf.keras.layers.Dense(1, name="head", activation="sigmoid") (x)
#     return tf.keras.models.Model(inputs=inputs, outputs=x, name="discriminator")

def define_discriminator(generate_what, n_filters = 32, blocks = 4, layers_per_block = 2):
    if generate_what == "elp_alone":
        sample_input = tf.keras.layers.Input((spe_width,1,1))
    elif generate_what == "all+annotations":
        sample_input = tf.keras.layers.Input((spe_width,2,6))
    x = sample_input
    
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
    
    discriminator = tf.keras.models.Model(sample_input, outputs, name = "discriminator")

    return discriminator

def define_generator(generate_what, latent_dim, generator_architecture, n_filters = 32, blocks = 4, layers_per_block = 2):
    latent_input = tf.keras.layers.Input(latent_dim)
    x = latent_input
    
    if generator_architecture == "conv":
        print("Using regular conv generator architecture")
        # reshape maps input using maxpooling
        maxpooling_factor = np.power(2, blocks)
        reduced_dim = spe_width // maxpooling_factor
        reduced_n_filters = n_filters * maxpooling_factor
        firstconv_dim = (reduced_dim, 1, reduced_n_filters)
    
        x = tf.keras.layers.Dense(np.prod(firstconv_dim), kernel_initializer='he_normal') (x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Reshape(firstconv_dim) (x)
        
        # run through u-net
        x = get_decoder(x, blocks=blocks, n_filters=n_filters, layers_per_block=layers_per_block)
    
    last_layer_activation = "sigmoid"
    print("Initializing generator with 'sigmoid' activation")
    
    if generate_what == 'elp_alone':
        final_output = tf.keras.layers.Conv2D(1, (1,1), activation=last_layer_activation) (x)
    elif generate_what == 'all+annotations':
        outputs_curves = tf.keras.layers.Conv2D(6, (1,1), activation=last_layer_activation) (x) # curves output
        outputs_maps = tf.keras.layers.Conv2D(6, (1,1), activation=last_layer_activation) (x) # maps output
        final_output = tf.keras.layers.Concatenate(axis=-2) ([outputs_curves, outputs_maps])
    
    generator = tf.keras.models.Model(latent_input, final_output, name = "generator")

    return generator

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_name):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        if loss_name == "wasserstein":            
            self.loss = "wasserstein"
            self.loss_fn = wasserstein_loss
        elif loss_name == "regular":
            self.loss = "regular"
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            assert False, "unknown loss function: {}".format(loss_name)
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        # for data in dataset.as_numpy_iterator():
        #     break
            
        # Unpack the data.
        batch_samples = data
        # and get batch size
        batch_size = tf.shape(batch_samples)[0]
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake samples using real maps
        generated_samples = self.generator(random_latent_vectors)

        # Combine them with real samples
        combined_samples = tf.concat([generated_samples, batch_samples], axis=0)

        # Assemble labels discriminating real from fake images
        if self.loss == "wasserstein":
            labels = tf.concat([tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0)
        elif self.loss == "regular":
            labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        else:
            assert False, "unknown loss: {}".format(self.loss)
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_samples)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        if self.loss == "wasserstein":
            misleading_labels = -tf.ones((batch_size, 1))
        elif self.loss == "regular":
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
    def __init__(self, path, num_img=3, every = 1, latent_dim=128):
        self.path = path
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every > 0:
            return
        # generate fake samples
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        # random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        generated_samples = self.model.generator.predict(random_latent_vectors)
        # save figures
        for i in range(self.num_img):
            plotSample(generated_samples[i,...], save = os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1)))
            plt.close()
    
# %%

# define HP
LATENT_DIM = FLAGS.latent_dim
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
MODEL_NAME = "transfull_it_v1"

# create folders
model_path = os.path.join(path_out,MODEL_NAME,"model")
imgs_path = os.path.join(path_out,MODEL_NAME,"imgs")
os.makedirs(imgs_path, exist_ok=True)

# instanciate models
discriminator = define_discriminator(generate_what = FLAGS.generate_what, n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)
# discriminator = define_discriminator_transformer(input_size = spe_width, hidden_size = 4, patch_size = 4, num_layers = 3, num_heads = 12, mlp_dim = 16, dropout=.1, representation_size = None, include_top=True)
generator = define_generator(generate_what = FLAGS.generate_what, latent_dim = LATENT_DIM, generator_architecture = FLAGS.generator_architecture, n_filters = FLAGS.n_filters, blocks = FLAGS.n_blocks, layers_per_block = FLAGS.n_layers_per_block)

discriminator.summary()
generator.summary()

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
    loss_name=FLAGS.loss,
)

callbacks = [
    GANMonitor(imgs_path, num_img=1, latent_dim=LATENT_DIM),
    tf.keras.callbacks.ModelCheckpoint(model_path+".h5", verbose=1, save_weights_only=False)
]

results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)

with open(model_path+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%
