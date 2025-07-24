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
# import tensorflow_addons as tfa

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--generator_weights", type=str, default=r"C:\Users\admin\Documents\Capillarys\data\delta_gan\out\spade_it_v2\model_pretrain.h5")

# FLAGS FOR GENERATOR HYPERPARAMETERS
# parser.add_argument("--generator_blocks", type=str, default= "256, 256, 256, 128,  64,  32")
# parser.add_argument("--generator_kernels", type=str, default="4-1, 4-1, 4-1, 4-1, 4-1, 4-1")
parser.add_argument("--generator_blocks", type=str, default= "128, 128, 128,  64,  32,  16")
parser.add_argument("--generator_kernels", type=str, default="3-1, 3-1, 3-1, 3-1, 3-1, 3-1")
parser.add_argument("--generator_learning_rate", type=float, default=1e-05)

# FLAGS FOR DISCRIMINATOR HYPERPARAMETERS
parser.add_argument("--discriminator_kernel", type=str, default="3-1")
parser.add_argument("--discriminator_dropout", type=float, default=.0)
parser.add_argument("--discriminator_learning_rate", type=float, default=1e-05)

# FLAGS FOR GENERAL HYPERPARAMETERS
parser.add_argument("--fadein_base_steps", type=int, default=20)
parser.add_argument("--fixed_base_steps", type=int, default=10)

FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
GENERATOR_BLOCKS = [int(k) for k in FLAGS.generator_blocks.split(",")]
GENERATOR_KERNELS = [[int(k) for k in sk.split("-")] for sk in FLAGS.generator_kernels.split(",")]
GENERATOR_LEARNING_RATE = FLAGS.generator_learning_rate
DISCRIMINATOR_KERNEL = [int(k) for k in FLAGS.discriminator_kernel.split("-")]
DISCRIMINATOR_DROPOUT = FLAGS.discriminator_dropout
DISCRIMINATOR_LEARNING_RATE = FLAGS.discriminator_learning_rate
VERBOSE = 2-(FLAGS.host=="local")*1
DEBUG = FLAGS.debug > 0
FADEIN_BASE_STEPS = FLAGS.fadein_base_steps
FIXED_BASE_STEPS = FLAGS.fixed_base_steps

if DEBUG:
    FADEIN_BASE_STEPS = 5
    FIXED_BASE_STEPS = 2

# reload pretrained generator

MODEL_NAME = "spade_it_v3"

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

if DEBUG:
    if_x = if_x[:16,...]
    if_y = if_y[:16,...]

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
        
        if crv_a.shape[0] != crv_v.shape[0]:
            downsample_factor = int(crv_a.shape[0]//crv_v.shape[0])
            crv_a = crv_a.reshape(-1, downsample_factor).mean(axis=1)
            crv_a_ref = crv_a_ref.reshape(-1, downsample_factor).mean(axis=1)
        
        plt.subplot(6,2,2*j+1)
        # fill area
        # plt.fill_between(crv_x[crv_a_ref==1], crv_v[crv_a_ref==1], color='#aaaaff')
        # plt.fill_between(crv_x[crv_a==1], crv_v[crv_a==1], color='#ffaaaa')
        # plot curves
        start_loc = 0
        start_a = int(np.round(crv_a_ref[start_loc]))
        ## fill in peaks
        start_loc = 0
        start_a = int(np.round(crv_a[start_loc]))
        cur_loc = 1
        while True:
            if cur_loc >= crv_x.shape[0]:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#ffaaaa')
                plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                break
            new_a = int(np.round(crv_a[cur_loc]))
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
            new_a = int(np.round(crv_a_ref[cur_loc]))
            if new_a != start_a:
                if start_a==1:
                    plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#aaaaff')
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[2])
                start_loc = cur_loc
                start_a = new_a
            cur_loc += 1
        ##
        plt.text(crv_v.shape[0],1,plot_names[j], color="#ffffff", verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle='square', facecolor='#000000', alpha=0.8))
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
        plt.tick_params(axis='y',       # changes apply to the x-axis and the y-axis
                        which='both',      # both major and minor ticks are affected
                        top=False,         # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelleft=False)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
        
def plotSampleWithRef(ref_curves, pred_curves, labels, save = None, noise_stddev = 0):
    from matplotlib import pyplot as plt
    
    plot_colors = ('black','red','blue')
    plot_names = ( ("Reference (true)","G (true)","A (true)","M (true)","k (true)","l (true)"), 
                   ("Reference (pred)","G (pred)","A (pred)","M (pred)","k (pred)","l (pred)")  )
            
    i = 0
    plt.figure(figsize=(8,8))
    for curves_column, curves in enumerate( (ref_curves, pred_curves) ):
        crv_x = np.arange(curves.shape[1])
        for j in range(6):
            crv_v = curves[i,:,0,j].copy()
            if noise_stddev > 0:
                crv_v += np.random.normal(loc=0, scale=noise_stddev, size=crv_v.shape)
            crv_a = (labels[i,:,0,j]>.5)*1
            if j==0:
                crv_a = 1-crv_a
            crv_a_ref = np.logical_and(crv_a==0,np.max(labels[i,:,0,1:],axis=-1)>.5)*1
            
            if crv_a.shape[0] != crv_v.shape[0]:
                downsample_factor = int(crv_a.shape[0]//crv_v.shape[0])
                crv_a = crv_a.reshape(-1, downsample_factor).mean(axis=1)
                crv_a_ref = crv_a_ref.reshape(-1, downsample_factor).mean(axis=1)
            
            plt.subplot(6,3,3*j+1+curves_column)
            # fill area
            # plt.fill_between(crv_x[crv_a_ref==1], crv_v[crv_a_ref==1], color='#aaaaff')
            # plt.fill_between(crv_x[crv_a==1], crv_v[crv_a==1], color='#ffaaaa')
            # plot curves
            start_loc = 0
            start_a = int(np.round(crv_a_ref[start_loc]))
            ## fill in peaks
            start_loc = 0
            start_a = int(np.round(crv_a[start_loc]))
            cur_loc = 1
            while True:
                if cur_loc >= crv_x.shape[0]:
                    if start_a==1:
                        plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#ffaaaa')
                    plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[start_a])
                    break
                new_a = int(np.round(crv_a[cur_loc]))
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
                new_a = int(np.round(crv_a_ref[cur_loc]))
                if new_a != start_a:
                    if start_a==1:
                        plt.fill_between(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color='#aaaaff')
                        plt.plot(crv_x[start_loc:cur_loc], crv_v[start_loc:cur_loc], color=plot_colors[2])
                    start_loc = cur_loc
                    start_a = new_a
                cur_loc += 1
            ##
            plt.text(crv_v.shape[0],1,plot_names[curves_column][j], color="#ffffff", verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle='square', facecolor='#000000', alpha=0.8))
            plt.tick_params(axis='both',       # changes apply to the x-axis and the y-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            left=False,
                            right=False,
                            labelbottom=False, # labels along the bottom edge are off
                            labelleft=False)
            plt.ylim(-.1, 1.1)
            if curves_column == 1:
                plt.subplot(6,3,3*j+3)
                if j==0:
                    plt.plot(crv_x, 1-crv_a, color="#333333")
                else:
                    plt.plot(crv_x, crv_a, color="#333333")
                plt.ylim(-.1, 1.1)
                plt.text(0,1,plot_names[curves_column][j], color="#ffffff", verticalalignment="top", horizontalalignment="left", bbox=dict(boxstyle='square', facecolor='#000000', alpha=0.8))
                plt.tick_params(axis='y',       # changes apply to the x-axis and the y-axis
                                which='both',      # both major and minor ticks are affected
                                top=False,         # ticks along the top edge are off
                                left=False,
                                right=False,
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

class WeightedSum(tf.keras.layers.Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')
 
	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2), "Len of inputs must be strictly equal to 2"
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

def spade_block(input_tensor, label_tensor, k, name, kernel_size):
    _, inputs_height, inputs_width, _ = input_tensor.shape
    
    # annotations are resized according to the input size
    # from tf 2.6.0,, resizing layer is included in the base tf.keras.layers
    if False:
        if tf.__version__ >='2.6.*':
            a = tf.keras.layers.Resizing(height=inputs_height, width=inputs_width,
                                          name = name + "_LabelResizing",
                                          crop_to_aspect_ratio=False,
                                          interpolation='nearest') (label_tensor)
        else:
            a = tf.keras.layers.experimental.preprocessing.Resizing(height=inputs_height, width=inputs_width,
                                                                    interpolation = "nearest",
                                                                    name = name + "_LabelResizing") (label_tensor)
    # TODO replaced by max pooling for keeping 1/0
    else:
        _, labels_height, labels_width, _ = label_tensor.shape
        a = tf.keras.layers.MaxPooling2D(pool_size=(labels_height//inputs_height,1)) (label_tensor)
    
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

def add_spade_generator_block(base_model, index, filters, kernel_size):
    # add block
    base_model_featuremaps = base_model.layers[-2].output
    base_model_featuremaps_upsampled = tf.keras.layers.UpSampling2D(size=(2,1), name="SpadeRes{}_up".format(index)) (base_model_featuremaps)
    new_block = spade_res_block(input_tensor=base_model_featuremaps_upsampled, label_tensor=base_model.inputs[1], k=filters, name="SpadeRes{}".format(index), kernel_size=kernel_size)
    # add final 1x1-conv and make model
    # add final 1x1-conv
    out_highres = tf.keras.layers.Conv2D(6, kernel_size=(1,1), padding="same", activation="sigmoid") (new_block)
    main_model = tf.keras.models.Model(base_model.inputs, [out_highres])
    # 
    out_upsampled_lowres = base_model.layers[-1] (base_model_featuremaps_upsampled)
    merged = WeightedSum() ([out_upsampled_lowres, out_highres]) # old, new
    growing_model = tf.keras.models.Model(base_model.inputs, [merged])
    
    return [main_model, growing_model] # main, then growing

def get_spade_generators(output_shape, blocks, kernel_sizes):
    assert len(blocks) == len(kernel_sizes), "Len of blocks must match len of kernel sizes"
    # compute input shape
    input_shape = output_shape[0]/np.power(2,len(blocks)-1)
    assert input_shape % 1 == 0, "First dim of label shape must be a multiple of the number of blocks (i.e., upsampling layers)"
    input_shape = [int(input_shape),] + list(output_shape[1:-1]) + [blocks[0],]
    
    # create inputs
    input_tensor = tf.keras.layers.Input(int(np.prod(input_shape)), name="Input_Noise")
    label_tensor = tf.keras.layers.Input(output_shape, name="Input_Labels")
    
    # reshape noise to conv shape
    x = tf.keras.layers.Reshape(target_shape=input_shape) (input_tensor)
    
    # add first block
    x = spade_res_block(input_tensor=x, label_tensor=label_tensor, k=blocks[0], name="SpadeRes{}".format(1), kernel_size=kernel_sizes[0])

    # add final 1x1-conv
    out = tf.keras.layers.Conv2D(6, kernel_size=(1,1), padding="same", activation="sigmoid") (x)
    
    # compute base model
    base_model = tf.keras.models.Model([input_tensor,label_tensor],[out])
    
    model_list = [[base_model,base_model]]
    
    # add blocks to submodels
    for i in range(1, len(blocks)):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_spade_generator_block(old_model, index=i+1, filters=blocks[i], kernel_size=kernel_sizes[i])
        # store model
        model_list.append(models)
        
    return model_list

def get_discriminator_inilayers(input_tensor, filters, name):
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_initializer="he_normal", name=name+"-1x1-conv") (x)
    x = tf.keras.layers.ReLU(name=name+"-relu") (x)
    return x

def get_discriminator_block(input_tensor, filters, kernel_size, dropout, name):
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1,1),
                               kernel_initializer="he_normal", padding="same",
                               name=name+"-conv") (x)
    if dropout>0.:
        x = tf.keras.layers.Dropout(rate = dropout, name=name+"-dropout") (x)
    x = tf.keras.layers.BatchNormalization(name=name+"-batchnorm") (x)
    # x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform", name=name+"-norm") (x)
    x = tf.keras.layers.LeakyReLU(name=name+"-lrelu") (x)
    return x

def add_discriminator_block(lower_resolution_model, index, filters, kernel_size, dropout, n_input_layers=3):
    # get shape of existing model
    in_shape = list(lower_resolution_model.input.shape)
    # define new input shape as double the size
    higher_resolution_input_shape = (in_shape[1]*2, 1, in_shape[-1])
    higher_resolution_input = tf.keras.layers.Input(shape=higher_resolution_input_shape, name="res-{}-input".format(in_shape[1]*2))
    # define new input processing layer
    x = get_discriminator_inilayers(higher_resolution_input, filters=filters, name="ini-{}".format(in_shape[1]*2))
    # define new block
    x = get_discriminator_block(x, filters=filters, kernel_size=kernel_size, dropout=dropout, name="block-{}".format(index))
    x = tf.keras.layers.AveragePooling2D((2,1), name="block-{}-pool".format(index)) (x)
    block_new = x
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(lower_resolution_model.layers)):
        x = lower_resolution_model.layers[i] (x)
    # define straight-through model
    highres_model = tf.keras.models.Model(higher_resolution_input, x)
    # in parallel ; downsample the new larger image, then feed the old model with it
    downsampled_input = tf.keras.layers.AveragePooling2D((2,1), name="block-{}-inputpool".format(index)) (higher_resolution_input)
    # connect old input processing to downsampled new input
    block_old = lower_resolution_model.layers[1] (downsampled_input)
    block_old = lower_resolution_model.layers[2] (block_old)
    # fade in output of old model input layer with new input
    x = WeightedSum() ([block_old, block_new]) # old, new
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(lower_resolution_model.layers)):
        x = lower_resolution_model.layers[i] (x)
    # define straight-through model
    growing_model = tf.keras.models.Model(higher_resolution_input, x)
    return [highres_model, growing_model]

def get_discriminators(input_shape, blocks, filters, kernel_size, dropout):
    initial_size = input_shape[0]/np.power(2,blocks-1)
    assert initial_size % 1 == 0, "input_shape[0]/(2^blocks) must be an integer"
    initial_size = int(initial_size)
    
    # apply initial block (always identical) : input -> 1x1-conv -> ReLU
    input_tensor = tf.keras.layers.Input((initial_size,) + input_shape[1:], name="res-{}-input".format(initial_size))
    x = get_discriminator_inilayers(input_tensor, filters=filters, name="ini-{}".format(initial_size))
    
    # add a block
    x = get_discriminator_block(x, filters=filters, kernel_size=kernel_size, dropout=dropout, name="block-1")
        
    # add output -> values between -1 and 1, 1 value by sample
    x = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="tanh", name="final-1x1-conv") (x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="finalpool") (x)
    
    base_model = tf.keras.models.Model(inputs=[input_tensor], outputs=[x])
    
	# store model
    model_list = [[base_model, base_model]]
    
    # create submodels
    for i in range(1, blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model,
                                         index=i+1,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         dropout=dropout,
                                         n_input_layers=3)
        # store model
        model_list.append(models)
        
    return model_list

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

class ProgressiveGrowingGAN(tf.keras.Model):
    def __init__(self, discriminators, generators):
        super(ProgressiveGrowingGAN, self).__init__()
        self.discriminators = discriminators
        self.generators = generators
        self.latent_dim = generators[0][0].inputs[0].shape[-1]
        self.step_resolution = 0
        self.step_alpha = 0.
        self.mode = 0 # start at fixed (not growing)
        
    def resetModels(self):
        self.generator = self.generators[self.step_resolution][self.mode]
        self.discriminator = self.discriminators[self.step_resolution][self.mode]
        self.d_optimizer = self.d_optimizers[self.step_resolution]
        self.g_optimizer = self.g_optimizers[self.step_resolution]
        # set alpha
        for model in [self.generator, self.discriminator]:
             for layer in model.layers:
                 if isinstance(layer, WeightedSum):
                     tf.keras.backend.set_value(layer.alpha, self.step_alpha)

    def compile(self, g_learning_rate, d_learning_rate):
        super(ProgressiveGrowingGAN, self).compile()
        self.d_optimizers = [tf.keras.optimizers.Adam(lr=d_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8) for i in range(len(self.discriminators))]
        self.g_optimizers = [tf.keras.optimizers.Adam(lr=d_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8) for i in range(len(self.generators))]
        self.d_loss_fn = discriminator_hinge_loss
        self.g_loss_fn = generator_hinge_loss
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.resetModels()

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    # def save(self):
    #     assert False, "Not coded yet"
    #     return

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
        
        # adapt resolution of real curves and labels to the resolution of the fake curves
        batch_curves_downsized = tf.image.resize(batch_curves, generated_curves.shape[1:-1])
        
        #batch_labels_downsized = tf.image.resize(batch_labels, generated_curves.shape[1:-1], method="nearest")
        batch_labels_downsized = tf.keras.layers.MaxPooling2D(pool_size=(batch_labels.shape[1] // generated_curves.shape[1], 1)) (batch_labels)

        # Combine them with real samples (real then fake)
        combined_curves = tf.concat([batch_curves_downsized, generated_curves], axis=0)
        combined_labels = tf.concat([batch_labels_downsized, batch_labels_downsized], axis=0)

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
            predictions = self.discriminator(tf.concat([combined_curves, combined_labels], axis=-1))
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
            predictions = self.discriminator( tf.concat([self.generator( [random_latent_vectors, batch_labels] ), batch_labels_downsized], axis=-1) )
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
    
class ExportExampleFigure(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, reference_data, path, export_every_n_epochs=1):
        self.path = path
        self.latent_dim = latent_dim
        self.reference_data = reference_data
        self.export_every_n_epochs = export_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.export_every_n_epochs > 0:
            return
        # generate random noise
        random_latent_vectors = tf.random.normal(shape=(self.reference_data.shape[0], self.latent_dim))
        # generate random labels
        reference_labels = self.reference_data[...,6:]
        # generate fake samples
        generated_samples = self.model.generator.predict([random_latent_vectors,reference_labels])
        # save figures
        for i in range(self.reference_data.shape[0]):
            if self.path is not None:
                save = os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1))
                plotSampleWithRef(self.reference_data[[i],...,:6], generated_samples[[i],...], reference_labels[[i],...], save = save)
                plt.close()
            else:
                save = None
                plotSampleWithRef(self.reference_data[[i],...,:6], generated_samples[[i],...], reference_labels[[i],...], save = save)
    
class AdjustWeightedSumAlpha(tf.keras.callbacks.Callback):
    def __init__(self, fadein_steps = 100, fixed_steps=50):
        self.fadein_steps = float(fadein_steps)
        self.fixed_steps = fixed_steps
        self.step = 0.
        # 1 for growing, 0 for fixed

    def on_epoch_end(self, epoch, logs=None):
        self.step += 1. # increase step
        if self.model.mode == 1: # growing
            new_alpha = self.step / (self.fadein_steps - 1) # compute new alpha
            if new_alpha > 1.: # go to fixed
                print("Going to fixed resolution mode for {:.0f} epochs".format(self.fixed_steps))
                self.fadein_steps *= 2
                self.step = 0.
                self.model.mode = 0 # go to fixed mode
                self.model.step_alpha = 1. # set alpha = 1 (high res)
                self.model.resetModels()
            else: # simply adjust alpha
                print("Setting alpha to {:.3f}".format(new_alpha))
                self.model.step_alpha = new_alpha
                self.model.resetModels()
        elif self.model.step_resolution < (len(self.model.generators)-1): # mode = fixed -> only act if over (go to growing)
            if self.step >= self.fixed_steps: # mode is over
                print("Going to growing mode for {:.0f} epochs".format(self.fadein_steps))
                self.fixed_steps *= 2
                self.step = 0.
                self.model.mode = 1 # go to growing
                self.model.step_alpha = 0. # reset alpha to 0 (low res)
                self.model.step_resolution += 1
                self.model.resetModels()
    
# %%

# create folders
MODEL_PATH = os.path.join(PATH_OUT,MODEL_NAME,"model")
IMGS_PATH = os.path.join(PATH_OUT,MODEL_NAME,"imgs")
# PRETRAIN_IMGS_PATH = os.path.join(PATH_OUT,MODEL_NAME,"pretrain_imgs")
os.makedirs(IMGS_PATH, exist_ok=True)
# os.makedirs(PRETRAIN_IMGS_PATH, exist_ok=True)

# instanciate models
generators = get_spade_generators(output_shape=(spe_width,1,6),
                                  blocks=GENERATOR_BLOCKS,
                                  kernel_sizes=GENERATOR_KERNELS)
discriminators = get_discriminators((spe_width,1,12),
                                    blocks=len(GENERATOR_BLOCKS),
                                    filters=64,
                                    kernel_size=DISCRIMINATOR_KERNEL,
                                    dropout=DISCRIMINATOR_DROPOUT)

LATENT_DIM = generators[0][0].inputs[0].shape[-1]

##### GAN TRAINING

gan = ProgressiveGrowingGAN(discriminators=discriminators, generators=generators)

gan.compile(g_learning_rate = GENERATOR_LEARNING_RATE,
            d_learning_rate = DISCRIMINATOR_LEARNING_RATE)

gan.generators[0][0].summary()
gan.discriminators[0][0].summary()

# data for callback
for batch in dataset.as_numpy_iterator():
    figure_ref_data = batch[[0],...]

callbacks = [ExportExampleFigure(latent_dim=LATENT_DIM, reference_data = figure_ref_data, path=IMGS_PATH, export_every_n_epochs=1),
             AdjustWeightedSumAlpha(fadein_steps = FADEIN_BASE_STEPS, fixed_steps = FIXED_BASE_STEPS)]
    
results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=VERBOSE)

with open(MODEL_PATH+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%
