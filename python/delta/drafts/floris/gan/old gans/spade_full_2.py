# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:58:26 2021

@author: admin
"""

# un gan pour créer les 6 courbes à partir d'annotations
# generator : input  = random noise (LATENT_DIM,) + 304x6x1 (annotations)
#             output = (304,6,1)

# TODO

# Mettre le discriminateur en 4,1 uniquement car 4,5 a l'air d'être compliqué pour lui... ? (ou alors est-ce uniquement
# par que le générateur devient trop bon quand on lui met des conv 4,5 ????)

# Donner des annotatinos sur albumine/a1/a2 etc pour délimiter les fractions ? car sinon le générateur
# met l'albumine ou il veut...
# IMPOSSIBLE car on a pas les annotations des fractions pour ces courbes
# on pourrait le faire en maths mais ce serait long, fastidieux et probablement erreurs ++

# TODO
# et si on faisait un 2e discriminateur qui reçoit uniquement les courbes stackées en channels (304x1x6)
# et son job a lui est de vérifier que les courbes se ressemblent ?

# TODO le bruit ne devrait pas être différent pour chaque dimension
# i.e. le bruit devrait être pour 304x1, et ensuite répliqué pour 304x1x6 !!!
# car sinon pas les mêmes données pour les différentes courbes

# possible aussi -> en utilisant qu'un seul disc, mais en lui donnant les courbes sous ce format ? (304x1x12 du coup)




# TODO peut être que le modèle n'y arrive pas car il a la majorité des zones avec que des 0
# peut être faudrait-il lui donner des 1 sur un des channels (là ou il n'y a aucun pic)
# pour lui dire ou mettre du "normal" ?
# en gros -> inverser la label de la courbe ELP (lui mettre 1,1,1,...,1,1,1,0,0,0,1,...,1,1)

# TODO rajouter des denses à la fin du discriminateur pour être sûr que l'albumine soit placée
# au début et pas à la fin
# i.e. pour supprimer l'invariance translationnelle qui existe pour le G et le D actuellement


# TODO
# Skip updates of the generator or the discriminator
# if either of them is becoming too strong.

# Stabilize with noise
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/





# TODO ajouter du noise à chaque couche du discriminateur ? voir comment faire...

# TODO pre-train generator with MSE for X epochs



# TODO custom save -> save generator and discriminator weights separately




# format 304x1x6 channels
# pareil pour annotations
# discriminateur : (304x1x6),(304x1x6), -> 304x1x12

import argparse
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
plt.ioff()

import tensorflow as tf
import tensorflow_addons as tfa

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--generator_weights", type=str, default=r"C:\Users\admin\Documents\Capillarys\data\delta_gan\out\spade_it_v2\model_pretrain.h5")

# FLAGS FOR GENERATOR HYPERPARAMETERS
parser.add_argument("--generator_blocks", type=str, default="1024,512,256,128,64,32")
parser.add_argument("--generator_kernels", type=str, default="4-1,4-1,4-1,4-1,4-1,4-1,4-1")
parser.add_argument("--generator_learning_rate", type=float, default=1e-04)

# FLAGS FOR DISCRIMINATOR HYPERPARAMETERS
parser.add_argument("--discriminator_blocks", type=str, default="32,64,128")
parser.add_argument("--discriminator_kernels", type=str, default="4-1,4-1,4-1,4-1")
parser.add_argument("--discriminator_dropout", type=float, default=.5)
parser.add_argument("--discriminator_learning_rate", type=float, default=1e-05)
parser.add_argument("--discriminator_noise_std", type=float, default=.01)
parser.add_argument("--discriminator_noise_decayfactor", type=float, default=.9)
parser.add_argument("--discriminator_noise_decayepochs", type=int, default=10)

# FLAGS FOR GENERAL HYPERPARAMETERS
parser.add_argument("--pretrain_epochs", type=int, default=0)
parser.add_argument("--pretrain_learning_rate", type=float, default=0.001)

FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

GENERATOR_BLOCKS = [int(k) for k in FLAGS.generator_blocks.split(",")]
DISCRIMINATOR_BLOCKS = [int(k) for k in FLAGS.discriminator_blocks.split(",")]
GENERATOR_KERNELS = [[int(k) for k in sk.split("-")] for sk in FLAGS.generator_kernels.split(",")]
DISCRIMINATOR_KERNELS = [[int(k) for k in sk.split("-")] for sk in FLAGS.discriminator_kernels.split(",")]
DISCRIMINATOR_DROPOUT = FLAGS.discriminator_dropout
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
GENERATOR_LEARNING_RATE = FLAGS.generator_learning_rate
DISCRIMINATOR_LEARNING_RATE = FLAGS.discriminator_learning_rate
DISCRIMINATOR_NOISE_STD = FLAGS.discriminator_noise_std
DISCRIMINATOR_NOISE_DECAYFACTOR = FLAGS.discriminator_noise_decayfactor
DISCRIMINATOR_NOISE_DECAYEPOCHS = FLAGS.discriminator_noise_decayepochs
PRETRAIN_LEARNING_RATE = FLAGS.pretrain_learning_rate
PRETRAIN_EPOCHS = FLAGS.pretrain_epochs
VERBOSE = 2-(FLAGS.host=="local")*1
GENERATOR_WEIGHTS_PATH = None
if len(FLAGS.generator_weights):
    if os.path.exists(FLAGS.generator_weights):
        GENERATOR_WEIGHTS_PATH = FLAGS.generator_weights

# reload pretrained generator

MODEL_NAME = "spade_it_v2"

if FLAGS.host=="local":
    PATH_IN = r'C:\Users\admin\Documents\Capillarys\data\2021\ifs'
    PATH_OUT = r'C:\Users\admin\Documents\Capillarys\data\delta_gan\out'
else:
    PATH_IN = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/data'
    PATH_OUT = '/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/out'
    
# %%

# C'est parti pour un modèle d'essai
if_x = np.load(os.path.join(PATH_IN,'if_v1_x.npy'))
if_y = np.load(os.path.join(PATH_IN,'if_v1_y.npy'))

spe_width = 320

# expand dims for X : n,304,6 -> n,304,6,1
if_x = np.expand_dims(if_x, axis=-2)

# expand dims for X : n,304,6 -> n,304,6,1
if_y = np.expand_dims(if_y, axis=-2)
# and concat "ref" curve for annotations
# on inverse la ref
if_y = np.concatenate([1-np.expand_dims(if_y.max(axis=-1), axis=-1), if_y], axis=-1)

# add padding in order to match spe_width
padding = spe_width-if_x.shape[1]
padding_start = padding//2
padding_end = padding - padding_start

if_x = np.concatenate([np.zeros([if_x.shape[0], padding_start, 1, 6]), if_x, np.zeros([if_x.shape[0], padding_end, 1, 6])], axis=1)
if_y = np.concatenate([np.zeros([if_y.shape[0], padding_start, 1, 6]), if_y, np.zeros([if_y.shape[0], padding_end, 1, 6])], axis=1)

# concat x and y for the generator
# dataset = tf.data.Dataset.from_tensor_slices( [ if_x.astype(np.float32), if_y.astype(np.float32) ] )
dataset = tf.data.Dataset.from_tensor_slices( np.concatenate([if_x,if_y],axis=-1).astype(np.float32) )
dataset = dataset.shuffle(buffer_size=256).batch(BATCH_SIZE)

# function for plotting a sample (used for the monitoring callback and for debugging the generator)
def plotSample(curves, labels, save = None, noise_stddev = 0):
    from matplotlib import pyplot as plt
    
    plot_colors = ('black','red','blue')
    plot_names = ("Reference","G","A","M","k","l")
    crv_x = np.arange(curves.shape[1])
            
    i = 0
    plt.figure(figsize=(8,8))
    for j in range(6):
        crv_v = curves[i,:,0,j].copy()
        if noise_stddev > 0:
            crv_v += np.random.normal(loc=0, scale=noise_stddev, size=crv_v.shape)
        crv_a = (labels[i,:,0,j]>.5)*1
        if j==0:
            crv_a = 1-crv_a
        crv_a_ref = np.logical_and(crv_a==0,np.max(labels[i,:,0,1:],axis=-1)>.5)*1
        plt.subplot(6,2,2*j+1)
        # fill area
        # plt.fill_between(crv_x[crv_a_ref==1], crv_v[crv_a_ref==1], color='#aaaaff')
        # plt.fill_between(crv_x[crv_a==1], crv_v[crv_a==1], color='#ffaaaa')
        # plot curves
        start_loc = 0
        start_a = crv_a_ref[start_loc]
        ## fill in peaks
        start_loc = 0
        start_a = crv_a[start_loc]
        cur_loc = 1
        while True:
            if cur_loc >= crv_x.shape[0]:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#ffaaaa')
                plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                break
            new_a = crv_a[cur_loc]
            if new_a != start_a:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#ffaaaa')
                plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                start_loc = cur_loc
                start_a = new_a
            cur_loc += 1
        ## fill in other peaks
        cur_loc = 1
        while True:
            if cur_loc >= crv_x.shape[0]:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#aaaaff')
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[2])
                break
            new_a = crv_a_ref[cur_loc]
            if new_a != start_a:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#aaaaff')
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[2])
                start_loc = cur_loc
                start_a = new_a
            cur_loc += 1
        ##
        plt.text(spe_width,1,plot_names[j], color="#ffffff", verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle='square', facecolor='#000000', alpha=0.8))
        plt.tick_params(axis='both',       # changes apply to the x-axis and the y-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelbottom=False, # labels along the bottom edge are off
                        labelleft=False)
        plt.ylim(-.1, 1.1)
        plt.subplot(6,2,j*2+2)
        if j==0:
            plt.plot(crv_x, 1-crv_a, color="#333333")
        else:
            plt.plot(crv_x, crv_a, color="#333333")
        plt.ylim(-.1, 1.1)
        plt.text(0,1,plot_names[j], color="#ffffff", verticalalignment="top", horizontalalignment="left", bbox=dict(boxstyle='square', facecolor='#000000', alpha=0.8))
        plt.tick_params(axis='both',       # changes apply to the x-axis and the y-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelbottom=False, # labels along the bottom edge are off
                        labelleft=False)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    
# function for generating random labels, based on the real data distribution
# first analyze the data distribution
mspike_label_params = {'Gk':dict(dims=[1,4]),
                       'Gl':dict(dims=[1,5]),
                       'Ak':dict(dims=[2,4]),
                       'Al':dict(dims=[2,5]),
                       'Mk':dict(dims=[3,4]),
                       'Ml':dict(dims=[3,5])}
for key,mspike_label_param in mspike_label_params.items():
    # 1. get labels concerning current dims
    labels_for_dims = np.sum(if_y[:,:,0,mspike_label_param['dims']], axis=-1)
    # 2. reject curves with multiple m-spikes -> will hamper proper calculation
    one_mspike_curves = np.where(np.sum(np.diff(labels_for_dims)>0, axis=1) == 1)[0]
    labels_for_dims = labels_for_dims[one_mspike_curves]
    # 3. filter only mspikes for those two dimensions
    labels_for_dims = (labels_for_dims==2)*1
    # 4. look for mspikes (label(c,p)==1)
    curve_index, pos_index = np.where(labels_for_dims==1)
    agg_df = pd.DataFrame(dict(curve_index=curve_index, pos_index=pos_index)).groupby(curve_index)
    spikes_centerpos = agg_df.mean() # -> list of center positions of m-spikes
    spikes_size = agg_df.count() # -> size (in pixels) of m-spikes
    mspike_label_param['center_mean']=spikes_centerpos.pos_index.mean()
    mspike_label_param['center_std']=spikes_centerpos.pos_index.std()
    mspike_label_param['size_mean']=spikes_size.pos_index.mean()
    mspike_label_param['size_std']=spikes_size.pos_index.std()
    
# now our function to generate fake labels
def generateFakeLabel(name="Gk", rng=np.random, mspike_label_params=mspike_label_params):
    fake_labels = np.concatenate([np.ones((1,spe_width,1,1)),np.zeros((1,spe_width,1,5))], axis=-1)
    center_pos = int(np.round(rng.normal(loc=mspike_label_params[name]['center_mean'], scale=mspike_label_params[name]['center_std'], size=1)[0]))
    size = int(np.round(rng.normal(loc=mspike_label_params[name]['size_mean'], scale=mspike_label_params[name]['size_std'], size=1)[0]))
    start_pos = center_pos-size//2
    end_pos = start_pos+size
    fake_labels[0,start_pos:end_pos,0,0] = 0
    for dim in mspike_label_params[name]['dims']:
        fake_labels[0,start_pos:end_pos,0,dim] = 1
    return fake_labels
    
if False:
    for batch in dataset.as_numpy_iterator():
        break
    
    def plotSamples(batch, noise_stddev):
        curves, labels = tf.split(batch, [6,6], axis=-1)
        curves, _ = tf.split(curves, [2, curves.shape[0]-2], axis=0)
        labels, _ = tf.split(labels, [2, labels.shape[0]-2], axis=0)
        curves = np.array(curves)
        labels = np.array(labels)
        for i in range(curves.shape[0]):
            plotSample(curves[[i],...], labels[[i],...], noise_stddev=noise_stddev)
    
    # 0.1 is huge
    # 0.01 is medium
    # 0.005 is really low
    plotSamples(batch, noise_stddev=.10)

# %%

def spade_block(input_tensor, label_tensor, k, name, kernel_size):
    _, inputs_height, inputs_width, _ = input_tensor.shape
    
    # annotations are resized according to the input size
    # from tf 2.6.0,, resizing layer is included in the base tf.keras.layers
    if tf.__version__ >='2.6.*':
        a = tf.keras.layers.Resizing(height=inputs_height, width=inputs_width,
                                     name = name + "_LabelResizing",
                                     crop_to_aspect_ratio=False,
                                     interpolation='nearest') (label_tensor)
    else:
        a = tf.keras.layers.experimental.preprocessing.Resizing(height=inputs_height, width=inputs_width,
                                                                interpolation = "nearest",
                                                                name = name + "_LabelResizing") (label_tensor)
    # then go through an interim conv layer followed by a ReLU activation
    interim_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, padding="same", name=name+"_ConvInterim") (a)
    interim_conv = tf.keras.layers.ReLU(name=name+"_ReLUInterim") (interim_conv)
    # then the feature maps output by this interim conv go in parallel through two different conv layers, one for producing the beta and one for the gamma
    mul_conv = tf.keras.layers.Conv2D(filters=k, kernel_size=kernel_size, padding="same", name=name+"_ConvMult") (interim_conv)
    add_conv = tf.keras.layers.Conv2D(filters=k, kernel_size=kernel_size, padding="same", name=name+"_ConvAdd") (interim_conv)

    # the input of the spade block goes through batch normalization
    x = tf.keras.layers.BatchNormalization(name = name + "_Batchnorm") (input_tensor)
    # and is multiplied by gamma and added to beta
    x = tf.keras.layers.Multiply(name=name+"_Mult") ([x, mul_conv])
    x = tf.keras.layers.Add(name=name+"_Add") ([x, add_conv])
    
    return x

def spade_res_block(input_tensor, label_tensor, k, name, kernel_size):
    ki = input_tensor.shape[-1]
    
    xL = input_tensor
    xR = input_tensor
    
    xL = spade_block(input_tensor=xL, label_tensor=label_tensor, k=ki, name=name+"_SubM1", kernel_size=[kernel_size[0],1])
    xL = tf.keras.layers.ReLU(name=name+"_SubM1_ReLU") (xL)
    xL = tf.keras.layers.Conv2D(k, kernel_size=kernel_size, padding="same", name=name+"_SubM1_Conv") (xL)
    xL = spade_block(input_tensor=xL, label_tensor=label_tensor, k=k, name=name+"_SubM2", kernel_size=[kernel_size[0],1])
    xL = tf.keras.layers.ReLU(name=name+"_SubM2_ReLU") (xL)
    xL = tf.keras.layers.Conv2D(k, kernel_size=kernel_size, padding="same", name=name+"_SubM2_Conv") (xL)
    
    xR = spade_block(input_tensor=xR, label_tensor=label_tensor, k=ki, name=name+"_SubS", kernel_size=[kernel_size[0],1])
    xR = tf.keras.layers.ReLU(name=name+"_SubS_ReLU") (xR)
    xR = tf.keras.layers.Conv2D(k, kernel_size=kernel_size, padding="same", name=name+"_SubS_Conv") (xR)
    
    x = tf.keras.layers.Add(name=name+"_ResAdd") ([xL, xR])
    
    return x

def get_spade_generator(label_shape, blocks, kernel_sizes):
    # compute input shape
    assert len(kernel_sizes) == len(blocks)+1, "Len of kernel sizes must be exactly len of blocks + 1 (for output layer)"
    input_shape = label_shape[0]/np.power(2,len(blocks))
    assert input_shape % 1 == 0, "First dim of label shape must be a multiple of the number of blocks (i.e., upsampling layers)"
    input_shape = [int(input_shape),]
    input_shape.extend([l for l in label_shape[1:-1]])
    input_shape.append(blocks[0])
    
    # create inputs
    input_tensor = tf.keras.layers.Input(int(np.prod(input_shape)), name="Input_Noise")
    label_tensor = tf.keras.layers.Input(label_shape, name="Input_Labels")
    
    x = tf.keras.layers.Reshape(target_shape=input_shape) (input_tensor)
    
    for block,k in enumerate(blocks):
        x = spade_res_block(input_tensor=x, label_tensor=label_tensor, k=k, name="SpadeRes{}".format(block+1), kernel_size=kernel_sizes[block])
        x = tf.keras.layers.UpSampling2D(size=(2,1), name="SpadeRes{}_up".format(block+1)) (x)
    x = tf.keras.layers.Conv2D(6, kernel_size=kernel_sizes[-1], padding="same", activation="sigmoid") (x)
    
    model = tf.keras.models.Model([input_tensor,label_tensor],[x])
    
    # model.summary()
    # if False:
    #     from tensorflow.keras.utils import plot_model 
    #     plot_model(model, to_file=r"C:\Users\admin\Documents\Capillarys\temp2021\temp.png", dpi=96)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=r"C:\Users\admin\Documents\Capillarys\temp2021\tb")
    # tensorboard.set_model(model) # your model here, will write graph etc
    # tensorboard.on_train_end() # will close the writer
    
    return model

def get_discriminator(input_shape, blocks, kernel_sizes, dropout):
    assert len(kernel_sizes) == len(blocks)+1, "Len of kernel sizes must be exactly len of blocks + 1 (for output layer)"
    input_tensor = tf.keras.layers.Input(input_shape, name="Input_Curves")
    label_tensor = tf.keras.layers.Input(input_shape, name="Input_Labels")
    
    # merge both on the channel dimension
    x = tf.keras.layers.Concatenate(axis=-1) ([input_tensor, label_tensor])
    
    for block,k in enumerate(blocks):
        strides = (2,1)
        x = tf.keras.layers.Conv2D(k, kernel_sizes[block], strides=strides, kernel_initializer="he_normal", padding="same") (x)
        if dropout>0.:
            x = tf.keras.layers.Dropout(rate = dropout) (x)
        if block>0:
            x = tfa.layers.InstanceNormalization(axis=3, 
                                                 center=True, 
                                                 scale=True,
                                                 beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform") (x)
        x = tf.keras.layers.LeakyReLU() (x)
        
    x = tf.keras.layers.Conv2D(1, kernel_sizes[-1], padding="same", activation="tanh") (x)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    # x = tf.keras.layers.Flatten() (x)
    # x = tf.keras.layers.Dense(1, activation="tanh") (x)
    
    model = tf.keras.models.Model(inputs=[input_tensor, label_tensor], outputs=[x])
    
    # model.summary()
    
    return model

def discriminator_hinge_loss(real, fake):
    # the discrminator should output ones (1) for real samples and -1 for fake samples
    # real = np.array([1.,1.,1.])
    # fake = np.array([-1.,-1.,-1.])
    # real = np.array([.9,.9,.9])
    # fake = np.array([-.9,-.9,-.9])
    return tf.maximum(tf.reduce_mean(1 - real), 0) + tf.maximum(tf.reduce_mean(1 + fake), 0)

def generator_hinge_loss(fake):
    # the generator should make the discriminator output only ones (1)
    return tf.maximum(tf.reduce_mean(- fake), 0)

class PreTrainer(tf.keras.Model):
    def __init__(self, generator):
        super(PreTrainer, self).__init__()
        self.generator = generator
        self.latent_dim = generator.inputs[0].shape[-1]

    def compile(self, optimizer):
        super(PreTrainer, self).compile()
        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.MAE
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_metric,]

    def train_step(self, data):
        # for data in dataset.as_numpy_iterator():
        #     break
    
        # Unpack the data.
        # batch_curves, batch_labels = data
        batch_curves, batch_labels = tf.split(data, [6,6], axis=-1)
        # and get batch size
        batch_size = tf.shape(batch_curves)[0]
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_curves = self.generator( [random_latent_vectors, batch_labels] )
            loss = self.loss_fn(generated_curves, batch_curves)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, noise_stddev):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = generator.inputs[0].shape[-1]
        self.noise_stddev = noise_stddev

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = discriminator_hinge_loss
        self.g_loss_fn = generator_hinge_loss
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        # for data in dataset.as_numpy_iterator():
        #     break
    
        # Unpack the data.
        # batch_curves, batch_labels = data
        batch_curves, batch_labels = tf.split(data, [6,6], axis=-1)
        # and get batch size
        batch_size = tf.shape(batch_curves)[0]
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake samples using real maps
        generated_curves = self.generator([random_latent_vectors, batch_labels])

        # Combine them with real samples (real then fake)
        combined_curves = tf.concat([batch_curves, generated_curves], axis=0)
        combined_labels = tf.concat([batch_labels, batch_labels], axis=0)

        # add noise to the samples
        if self.noise_stddev > 0:
            combined_curves += tf.random.normal(tf.shape(combined_curves), stddev=self.noise_stddev)
    
        # Associate labels: +1 for real, -1 for fakes (for discriminator)
        # discriminator_expected_output = tf.concat([tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        # discriminator_expected_output += 0.05 * tf.random.uniform(tf.shape(discriminator_expected_output))
        
        # for the discriminator, we need to add the labels at the output
        
        # print("batch size is: {}".format(batch_size))
        # print("combined_curves shape: {}".format(combined_curves.shape))
        # print("combined_labels shape: {}".format(combined_labels.shape))
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_curves, combined_labels])
            predictions_real, predictions_fake = tf.split(predictions, [batch_size, batch_size], axis=0)
            d_loss = self.d_loss_fn(real = predictions_real, fake = predictions_fake)
            
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            # predictions = self.discriminator( [self.generator( [random_latent_vectors, batch_labels] ), batch_labels] )
            # add noise to the samples
            generated_samples = self.generator( [random_latent_vectors, batch_labels] )
            if self.noise_stddev > 0:
                generated_samples += tf.random.normal(tf.shape(generated_samples), stddev=self.noise_stddev)
            predictions = self.discriminator( [generated_samples, batch_labels] )
            g_loss = self.g_loss_fn(fake = predictions)
            
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
    def __init__(self, latent_dim, path, num_img=3, export_every_n_epochs=1):
        self.path = path
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.export_every_n_epochs = export_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.export_every_n_epochs > 0:
            return
        # generate random noise
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        # generate random labels
        random_labels = np.concatenate([generateFakeLabel(name="Gk",rng=np.random.RandomState(0)) for i in range(self.num_img)])
        # generate fake samples
        generated_samples = self.model.generator.predict([random_latent_vectors,random_labels])
        # save figures
        for i in range(self.num_img):
            if self.path is not None:
                save = os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1))
                plotSample(generated_samples[[i],...], random_labels[[i],...], save = save)
                plt.close()
            else:
                save = None
                plotSample(generated_samples[[i],...], random_labels[[i],...], save = save)
    
class NoiseMonitor(tf.keras.callbacks.Callback):
    def __init__(self, decay_factor = .9, decay_epochs = 10):
        assert decay_factor <= 1., "Decay factor should be less or equal to 1"
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch > 0) & (epoch % self.decay_epochs == 0):
            old_noise_stddev = self.model.noise_stddev
            self.model.noise_stddev *= self.decay_factor
            print("Decaying noise stddev from {} to {}".format(old_noise_stddev,self.model.noise_stddev))
    
# %%

# create folders
MODEL_PATH = os.path.join(PATH_OUT,MODEL_NAME,"model")
IMGS_PATH = os.path.join(PATH_OUT,MODEL_NAME,"imgs")
PRETRAIN_IMGS_PATH = os.path.join(PATH_OUT,MODEL_NAME,"pretrain_imgs")
os.makedirs(IMGS_PATH, exist_ok=True)
os.makedirs(PRETRAIN_IMGS_PATH, exist_ok=True)

# instanciate models
generator = get_spade_generator(label_shape=(spe_width,1,6), blocks=GENERATOR_BLOCKS, kernel_sizes=GENERATOR_KERNELS)
discriminator = get_discriminator(input_shape=(spe_width,1,6), blocks=DISCRIMINATOR_BLOCKS, kernel_sizes=DISCRIMINATOR_KERNELS, dropout=DISCRIMINATOR_DROPOUT)

LATENT_DIM = generator.inputs[0].shape[-1]

# RELOAD PRETRAINED WEIGHTS
if GENERATOR_WEIGHTS_PATH is not None:
    
    # TODO
    
    
    print("Reloading generator weights from file: {}".format(GENERATOR_WEIGHTS_PATH))
    pretrainer = PreTrainer(generator)
    fake_curves = pretrainer.predict([np.random.normal(size=(1,generator.inputs[0].shape[-1])),
                                      generateFakeLabel(name="Gk",rng=np.random.RandomState(0))])
    pretrainer.load_weights(filepath=GENERATOR_WEIGHTS_PATH)

##### PRE-TRAIN

if PRETRAIN_EPOCHS > 0:
    print("Pretraining for {} epochs".format(PRETRAIN_EPOCHS))
    
    pretrainer = PreTrainer(generator)
    pretrainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=PRETRAIN_LEARNING_RATE))
    
    callbacks = [GANMonitor(latent_dim=LATENT_DIM, path=PRETRAIN_IMGS_PATH, num_img=1, export_every_n_epochs=1),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=.1, patience=10, verbose=VERBOSE),
                 tf.keras.callbacks.ModelCheckpoint(MODEL_PATH+"_pretrain.h5", verbose=VERBOSE, save_weights_only=True)]
    
    pretrainer.fit(dataset,
                   batch_size=BATCH_SIZE,
                   epochs=PRETRAIN_EPOCHS,
                   callbacks = callbacks,
                   verbose=VERBOSE)

##### GAN TRAINING

gan = GAN(discriminator=discriminator, generator=generator,
          noise_stddev=DISCRIMINATOR_NOISE_STD)

gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LEARNING_RATE),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=GENERATOR_LEARNING_RATE))

callbacks = [GANMonitor(latent_dim=LATENT_DIM, path=IMGS_PATH, num_img=1, export_every_n_epochs=1),
             tf.keras.callbacks.ModelCheckpoint(MODEL_PATH+".h5", verbose=VERBOSE, save_weights_only=True)]
if DISCRIMINATOR_NOISE_STD>0:
    callbacks.append(NoiseMonitor(decay_factor=DISCRIMINATOR_NOISE_DECAYFACTOR, decay_epochs=DISCRIMINATOR_NOISE_DECAYEPOCHS))

if False:    
    for data in dataset.as_numpy_iterator():
        break
    # batch_curves, batch_labels = data
    batch_curves, batch_labels = tf.split(data, [6,6], axis=-1)
    
    fake_curves = generator([np.random.normal(size=(8,generator.inputs[0].shape[-1])),batch_labels])
    
    plotSample(np.array(batch_curves)[[0]], np.array(batch_labels)[[0]])
    plotSample(np.array(fake_curves)[[0]], np.array(batch_labels)[[0]])
    
    discriminator([batch_curves,batch_labels])
    discriminator([generator([np.random.normal(size=(8,generator.inputs[0].shape[-1])),batch_labels]),batch_labels])
    
    cb = GANMonitor(latent_dim=LATENT_DIM, path=None)
    cb.model = gan
    cb.on_epoch_end(epoch=0)
    
results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=VERBOSE)

with open(MODEL_PATH+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%
