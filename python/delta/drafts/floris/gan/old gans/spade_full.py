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

# FLAGS FOR GENERATOR HYPERPARAMETERS
parser.add_argument("--generator_blocks", type=str, default="256,128,64,32")
parser.add_argument("--generator_kernels", type=str, default="4-5,4-5,4-5,4-5,4-5")

# FLAGS FOR DISCRIMINATOR HYPERPARAMETERS
parser.add_argument("--discriminator_blocks", type=str, default="32,64,128")
parser.add_argument("--discriminator_kernels", type=str, default="4-1,4-1,4-1,4-1")

# FLAGS FOR GENERAL HYPERPARAMETERS
parser.add_argument("--learning_rate", type=float, default=1e-04)

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
EPOCHS = 1000  # In practice, use ~100 epochs
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate

MODEL_NAME = "spade_it_v1"

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

spe_width = 304

# expand dims for X : n,304,6 -> n,304,6,1
if_x = np.expand_dims(if_x, axis=-1)

# expand dims for X : n,304,6 -> n,304,6,1
if_y = np.expand_dims(if_y, axis=-1)
# and concat "ref" curve for annotations
if_y = np.concatenate([np.expand_dims(if_y.max(axis=-2), axis=-1), if_y], axis=-2)

# concat x and y for the generator
# dataset = tf.data.Dataset.from_tensor_slices( [ if_x.astype(np.float32), if_y.astype(np.float32) ] )
dataset = tf.data.Dataset.from_tensor_slices( np.concatenate([if_x,if_y],axis=-1).astype(np.float32) )
dataset = dataset.shuffle(buffer_size=256).batch(BATCH_SIZE)

# function for plotting a sample (used for the monitoring callback and for debugging the generator)
def plotSample(curves, labels, save = None):
    from matplotlib import pyplot as plt
    
    plot_colors = ('black','red',)
    plot_names = ("Reference","G","A","M","k","l")
    crv_x = np.arange(curves.shape[1])
            
    i = 0
    plt.figure(figsize=(8,8))
    for j in range(6):
        crv_v = curves[i,:,j,0]
        crv_a = (labels[i,:,j,0]>.5)*1
        plt.subplot(6,2,2*j+1)
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
    labels_for_dims = np.sum(if_y[:,:,mspike_label_param['dims'],0], axis=-1)
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
    fake_labels = np.zeros((1,304,6,1))
    center_pos = int(np.round(rng.normal(loc=mspike_label_params[name]['center_mean'], scale=mspike_label_params[name]['center_std'], size=1)[0]))
    size = int(np.round(rng.normal(loc=mspike_label_params[name]['size_mean'], scale=mspike_label_params[name]['size_std'], size=1)[0]))
    start_pos = center_pos-size//2
    end_pos = start_pos+size
    for dim in mspike_label_params[name]['dims']:
        fake_labels[0,start_pos:end_pos,dim,0] = 1
    return fake_labels
    
if False:
    for batch in dataset.as_numpy_iterator():
        break
    
    def plotSamples(batch):
        curves, labels = tf.split(batch, [1,1], axis=-1)
        curves, _ = tf.split(curves, [2, curves.shape[0]-2], axis=0)
        labels, _ = tf.split(labels, [2, labels.shape[0]-2], axis=0)
        curves = np.array(curves)
        labels = np.array(curves)
        for i in range(curves.shape[0]):
            plotSample(curves[[i],...], labels[[i],...])
    
    plotSamples(batch)

# %%

def spade_block(input_tensor, label_tensor, k, name, kernel_size=(3,1)):
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

def spade_res_block(input_tensor, label_tensor, k, name, kernel_size=(3,1)):
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

def get_spade_generator(label_shape=(304,6,1), blocks=[128,64,32], kernel_sizes=[(3,1),(3,1),(3,1),(3,1)]):
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
    x = tf.keras.layers.Conv2D(1, kernel_size=kernel_sizes[-1], padding="same", activation="sigmoid") (x)
    
    model = tf.keras.models.Model([input_tensor,label_tensor],[x])
    
    # model.summary()
    # if False:
    #     from tensorflow.keras.utils import plot_model 
    #     plot_model(model, to_file=r"C:\Users\admin\Documents\Capillarys\temp2021\temp.png", dpi=96)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=r"C:\Users\admin\Documents\Capillarys\temp2021\tb")
    # tensorboard.set_model(model) # your model here, will write graph etc
    # tensorboard.on_train_end() # will close the writer
    
    return model

def get_discriminator(input_shape=(304,6,1), blocks=[32,64,128], kernel_sizes=[(3,1),(3,1),(3,1),(3,1)]):
    assert len(kernel_sizes) == len(blocks)+1, "Len of kernel sizes must be exactly len of blocks + 1 (for output layer)"
    input_tensor = tf.keras.layers.Input(input_shape, name="Input_Curves")
    label_tensor = tf.keras.layers.Input(input_shape, name="Input_Labels")
    
    # merge both on the channel dimension
    x = tf.keras.layers.Concatenate(axis=-1) ([input_tensor, label_tensor])
    
    for block,k in enumerate(blocks):
        strides = (2,1)
        x = tf.keras.layers.Conv2D(k, kernel_sizes[block], strides=strides, padding="same") (x)
        if block>0:
            x = tfa.layers.InstanceNormalization(axis=3, 
                                                 center=True, 
                                                 scale=True,
                                                 beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform") (x)
        x = tf.keras.layers.LeakyReLU() (x)
        
    # x = tf.keras.layers.Conv2D(1, (3,1), padding="same") (x)
    x = tf.keras.layers.Conv2D(1, kernel_sizes[-1], padding="same", activation="tanh") (x)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    
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

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = generator.inputs[0].shape[-1]

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
        batch_curves, batch_labels = tf.split(data, [1,1], axis=-1)
        # and get batch size
        batch_size = tf.shape(batch_curves)[0]
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake samples using real maps
        generated_curves = self.generator([random_latent_vectors, batch_labels])

        # Combine them with real samples (real then fake)
        combined_curves = tf.concat([batch_curves, generated_curves], axis=0)
        combined_labels = tf.concat([batch_labels, batch_labels], axis=0)

        # Associate labels: +1 for real, -1 for fakes (for discriminator)
        discriminator_expected_output = tf.concat([tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        discriminator_expected_output += 0.05 * tf.random.uniform(tf.shape(discriminator_expected_output))
        
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
            predictions = self.discriminator( [self.generator( [random_latent_vectors, batch_labels] ), batch_labels] )
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
            plotSample(generated_samples[[i],...], random_labels[[i],...], save = os.path.join(self.path,"generated_epoch{:04d}-{:02d}.jpg".format(epoch,i+1)))
            plt.close()
    
# %%

# create folders
MODEL_PATH = os.path.join(PATH_OUT,MODEL_NAME,"model")
IMGS_PATH = os.path.join(PATH_OUT,MODEL_NAME,"imgs")
os.makedirs(IMGS_PATH, exist_ok=True)

# instanciate models
generator = get_spade_generator(label_shape=(304,6,1), blocks=GENERATOR_BLOCKS, kernel_sizes=GENERATOR_KERNELS)
discriminator = get_discriminator(input_shape=(304,6,1), blocks=DISCRIMINATOR_BLOCKS, kernel_sizes=DISCRIMINATOR_KERNELS)

LATENT_DIM = generator.inputs[0].shape[-1]

gan = GAN(discriminator=discriminator, generator=generator)

gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

callbacks = [GANMonitor(latent_dim=LATENT_DIM, path=IMGS_PATH, num_img=1),
             tf.keras.callbacks.ModelCheckpoint(MODEL_PATH+".h5", verbose=1, save_weights_only=True)]

if False:    
    for data in dataset.as_numpy_iterator():
        break
    # batch_curves, batch_labels = data
    batch_curves, batch_labels = tf.split(data, [1,1], axis=-1)
    
    generator([np.random.normal(size=(8,generator.inputs[0].shape[-1])),batch_labels])
    discriminator([batch_curves,batch_labels])
    discriminator([generator([np.random.normal(size=(8,generator.inputs[0].shape[-1])),batch_labels]),batch_labels])
    
    # tmp = discriminator([batch_curves,batch_labels])
    
results = gan.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)

with open(MODEL_PATH+".log", 'wb') as file_pi:
    pickle.dump(results.history, file_pi)
    
# %%
