"""#############################################################################
#################### SPE IT MODEL TRAINING #####################################
author : Chabrun Floris and Dieu Xavier
date : 22/11/2024
Training segmentation models for immunosubtraction data
#############################################################################"""



"""=============================================================================
Imports
============================================================================="""

# general modules
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Union, Any
import math
from dataclasses import dataclass, fields
from collections import OrderedDict

# custom modules
from spep_assets.spep_data import ISDataset
from spep_assets.spep_figures import pp_size, plot_roc
from spep_assets.spep_stats import get_bootstrap_metric_ci

# PYTORCH
import torch
import torch.utils.data as data
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


#%%
"""=============================================================================
Paths and data loading
============================================================================="""

root_data_path = None
if "flori" in os.listdir(r"C:\Users"):
    root_data_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\data\proc\lemans_2018"
    output_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\output"
elif "afors" in os.listdir(r"C:\Users"):
    root_data_path = r"C:\Users\afors\Documents\Projects\SPE_IT\lemans_2018"
    # where to write everything
    output_path = r"C:\Users\afors\Documents\Projects\SPE_IT\output"
assert root_data_path is not None, "Unknown user"

# load Le Mans data

# load x array
# data is already normalized between 0-1 and zero-padded to a 304 width
if_x = np.load(r"C:\Users\afors\Documents\Projects\SPE_IT\lemans_2018\if_v1_x.npy")

# load y array
if_y = np.load(r"C:\Users\afors\Documents\Projects\SPE_IT\lemans_2018\if_v1_y.npy")

# note: we should create .h5 files to easily load the data using a data manager if we have a lot of samples!

# using a permutation of the subtraction trace to augment the dataset
# by getting all the combination of subtraction possible from one sample's five tracks
permutation_aug = True
if permutation_aug :
    def faster_permutations(n):
        # empty() is fast because it does not initialize the values of the array
        # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
        perms = np.empty((math.factorial(n), n), dtype=np.uint8, order='F')
        perms[0, 0] = 0

        rows_to_copy = 1
        for i in range(1, n):
            perms[:rows_to_copy, i] = i
            for j in range(1, i + 1):
                start_row = rows_to_copy * j
                end_row = rows_to_copy * (j + 1)
                splitter = i - j
                perms[start_row: end_row, splitter] = i
                perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
                perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side

            rows_to_copy *= i + 1

        return perms

    perms = faster_permutations(5) # we will permute all but the first reference track
    zeros = np.zeros((120,1), dtype=np.uint8)
    perms0 = np.hstack((zeros, perms+1))

    num_samples = if_x.shape[0]
    emptyArray_x = np.concatenate([np.zeros((1,304,6)) for i in range(num_samples*119)])
    emptyArray_y = np.concatenate([np.zeros((1,304,5)) for i in range(num_samples*119)])
    if_x = np.concatenate([if_x, emptyArray_x])
    if_y = np.concatenate([if_y, emptyArray_y])
    counter = num_samples
    for sample in range(num_samples) :
        print("permuting traces for patient : ", sample, "/", num_samples)
        for i in range(1, perms.shape[0]) :  # we don't need to add the first one since it is the original one
            if_x[counter] = np.expand_dims(np.transpose(if_x[sample, : , perms0[i]]), axis=0)
            if_y[counter] = np.expand_dims(np.transpose(if_y[0, :, perms[i]]), axis=0)
            counter+=1

debug_plots = True

if debug_plots:
    # show the firts sample of the dataset
    is_tracks = ["ELP", "IgG", "IgA", "IgM", "K", "L"]

    i = 0
    plt.figure(figsize=(12, 12))
    for j in range(6):
        plt.subplot(6, 2, j * 2 + 1)
        sns.lineplot(x=np.arange(304), y=if_x[i, :, j])  # plot data
        if j > 0:  # plot annotation map
            plt.subplot(6, 2, j * 2 + 2)
            sns.lineplot(x=np.arange(304), y=if_y[i, :, j - 1])
    plt.tight_layout()
    plt.show()


# %%
"""=============================================================================
Data splitting and dataloader setup
============================================================================="""

# we'll output the proportion of each class in the dataset (so we'll check if random partitioning works fine)
if debug_plots:
    print('IgG% : ', round(if_y[..., 0].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)),2), '\n',
          'IgA% : ', round(if_y[..., 1].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)),2), '\n',
          'IgM% : ', round(if_y[..., 2].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)),2), '\n',
          'Kappa% : ', round(if_y[..., 3].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)),2), '\n',
          'Lambda% : ', round(if_y[..., 4].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2))


# partition
if_x_train, if_x_test, if_y_train, if_y_test = train_test_split(if_x, if_y, test_size=.2, random_state=1, shuffle=True,
                                                                                      )

if debug_plots:
    print('IgG% train : ', round(if_y_train[..., 0].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)),2), '\n',
          'IgA% train : ', round(if_y_train[..., 1].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)),2), '\n',
          'IgM% train : ', round(if_y_train[..., 2].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)),2), '\n',
          'Kappa% train : ', round(if_y_train[..., 3].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)),2), '\n',
          'Lambda% train : ', round(if_y_train[..., 4].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2))

    print('IgG% test : ', round(if_y_test[..., 0].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)),2), '\n',
          'IgA% test : ', round(if_y_test[..., 1].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)),2), '\n',
          'IgM% test : ', round(if_y_test[..., 2].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)),2), '\n',
          'Kappa% test : ', round(if_y_test[..., 3].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)),2), '\n',
          'Lambda% test : ', round(if_y_test[..., 4].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2))

# seems well stratified

train_dataset = ISDataset(if_x=if_x_train, if_y=if_y_train, smoothing=False, normalize=False, coarse_dropout=False)
test_dataset = ISDataset(if_x=if_x_test, if_y=if_y_test, smoothing=False, normalize=False, coarse_dropout=False)

num_workers = 8  # how many processes will load data in parallel; 0 for none

# create our dataset loader for train data
train_loader = data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False
    # if we set >1 loader, we want them to be persistent, i.e. not being instantiated again between each epoch
)

# create our dataset loader for val data
validation_loader = data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False
)

if debug_plots:
    # just to check if the loader works OK
    first_train_batch = next(iter(train_loader))
    x, y = first_train_batch
    print(f"First training batch: {x.shape=} / {y.shape=}")

# %%
"""=============================================================================
Model instantiation
============================================================================="""

# TODO pré-entraîner modèle sur une autre tâche
# TODO masquer les pics pour forcer le modèle à regarder autour de l'albumine (expliquer à Xavier!)

# TODO note => SupervisedModule works OK
# TODO other models in spep_dl => they worked at some point, but a hell lot of modifications were made in between so really not so sure right now

# TODO instead of using grouped convolutions with 1D model, we may want to try using regular convolutions with a 2D model? (1 channel, but height = 6)
# TODO limit with this solution => 2nd dimension will move over "channels" (IgG/A/M/k/l) always in the same order so consider that "some IgG pattern" close to some "IgA pattern" and not "IgG" right before "Kappa" for instance... so maybe not the best solution?

# TODO segmentation instead of classification


class SegformerConfig:
    r"""
    This is the configuration class to store the configuration of a [`SegformerModel`]. It is used to instantiate an
    SegFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SegFormer
    [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            The number of layers in each encoder block.
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            Dimension of each of the encoder blocks.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            Patch size before each encoder block.
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            Stride before each encoder block.
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability before the classification head.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the all-MLP decode head.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    ```"""

    model_type = "segformer"

    def __init__(
        self,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
        num_labels = 0,
        use_return_dict = True,
        output_hidden_states = True,
        output_attentions = False
        ,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.num_labels = num_labels
        self.use_return_dict = use_return_dict
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions



class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, torch.Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())



@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class SemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Segformer
class SegformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv1d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        length = pixel_values.shape[1]
        pixel_values = pixel_values.transpose(1, 2)
        embeddings = self.proj(pixel_values)
        #_, _, length = embeddings.shape
        # base impl : (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
        # but we already have a sequence without height and width so we don't need to apply  flatten(2) to embeddings
        # this can be fed to a Transformer layer
        embeddings = embeddings.transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, length


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width) so we don't need .reshape(batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1)
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # No need to reshape back to (batch_size, seq_len, num_channels) so no .reshape(batch_size, num_channels, -1) before permute
            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio = 1):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)


    def forward(self, hidden_states, output_attentions):
        self_outputs = self.self(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SegformerDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, length):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, length):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, length)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, mlp_ratio, sequence_reduction_ratio = 1):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, length, output_attentions):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), length)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class SegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        stage_1_patch=False
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None


        #batch_size = pixel_values.shape[0]

        hidden_states = pixel_values

        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, length = embedding_layer(hidden_states)
            print(length)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, length, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            #if idx != len(self.patch_embeddings) - 1 or (
            #        idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            #):
            #    hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1,
            #                                                                                 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class SegformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states = hidden_states.flatten(2).transpose(1, 2)  not needed since we're in 1d already
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv1d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv1d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            #if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
            #    height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
            #    encoder_hidden_state = (
            #        encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            #    )

            # unify channel dimension
            length = encoder_hidden_state.shape[1]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            #encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, length)
            # upsample
            #print(encoder_hidden_state.shape)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[1], mode="linear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
            #print([all_hidden_states[i].shape for i in range(len(all_hidden_states))])

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        #print(hidden_states.shape)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #print(hidden_states.shape)
        # logits are of shape (batch_size, num_labels, length)
        logits = self.classifier(hidden_states)

        return logits


class IsSegformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        pixel_values = pixel_values.squeeze(dim=2).transpose(1,2)
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        #print([outputs.hidden_states[i].shape for i in range(len(outputs.hidden_states))])
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size => not needed here
            #upsampled_logits = nn.functional.interpolate(
            #    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            #)
            if self.config.num_labels > 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.transpose(1,2))
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )



# %%
"""=============================================================================
Model training
============================================================================="""

class pl_IS_model(pl.LightningModule):
    def __init__(self, model,
                 config,
                 optimizer="AdamW",
                 lr_scheduler="cosine_with_restarts",
                 lr=1e-4,
                 lr_reduceonplateau_factor=.5,
                 lr_reduceonplateau_patience=3,
                 lr_reduceonplateau_threshold=1e-2,
                 lr_reduceonplateau_minlr=1e-6,
                 num_warmup_steps=20,
                 num_training_steps=9000,
                 adam_eps = 1e-08,
                 weight_decay = 0.0
                 ):
        super().__init__()
        self.config = config
        self.model = model(self.config)
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.lr_reduceonplateau_factor = lr_reduceonplateau_factor
        self.lr_reduceonplateau_patience = lr_reduceonplateau_patience
        self.lr_reduceonplateau_threshold = lr_reduceonplateau_threshold
        self.lr_minlr = lr_reduceonplateau_minlr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        self.y_true = []
        self.y_pred = []


    def forward(self, batch):
        if type(batch) in (tuple, list):
            x, y = batch
        else:
            x = batch
        outputs = self.model(x)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x, y)
        #self.log("train_loss", outputs.loss)
        y_true=[]
        y_pred=[]
        #print([outputs.logits[i].shape for i in range(4)], [outputs.loss[i].shape for i in range(4)], [outputs.hidden_states[i].shape for i in range(4)])
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_outputs = self.model(x)

        #print(y.shape, val_outputs.logits.shape)
        self.y_true.append(y.transpose(1,2))
        self.y_pred.append(val_outputs.logits)
        #print([self.y_true[i].shape for i in range(len(self.y_true))], [self.y_pred[i].shape for i in range(len(self.y_pred))])

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        #print(y_true.shape, y_pred.shape)
        val_loss = nn.BCEWithLogitsLoss()
        ce_loss = val_loss(y_pred, y_true)
        #accuracy = torch.sum(torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)) / y_true.shape[0]

        log_dict = {'val_loss': ce_loss}
        #for i in range(self.n_classes):
        #    preds = torch.argmax(y_pred[torch.argmax(y_true, dim=1) == i], dim=1)
        #    log_dict[f"val_accuracy_class_{i}"] = torch.sum(preds == i) / preds.shape[0]

        self.log_dict(log_dict)

        self.y_true.clear()
        self.y_pred.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "RMSProp":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        else:
            assert False, f"Unknown {self.hparams.optimizer=}"

        if self.hparams.lr_scheduler == "reduceonplateau":
            print(
                f"Setting optimizer to reduceonplateau with params {self.hparams.lr_reduceonplateau_factor=}, {self.hparams.lr_reduceonplateau_patience=}, {self.hparams.lr_reduceonplateau_threshold=}, {self.hparams.lr_reduceonplateau_minlr=}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=self.hparams.lr_reduceonplateau_factor,
                                                                   patience=self.hparams.lr_reduceonplateau_patience,
                                                                   threshold=self.hparams.lr_reduceonplateau_threshold,
                                                                   min_lr=self.hparams.lr_reduceonplateau_minlr,
                                                                   verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        elif self.hparams.lr_scheduler == "multistep":
            print(
                f"Setting optimizer to multistep with params {self.hparams.lr_multistep_milestones=}, {self.hparams.lr_multistep_gamma=}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_multistep_milestones,
                                                             gamma=self.hparams.lr_multistep_gamma,
                                                             verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif self.hparams.lr_scheduler == "cosine_with_restarts":
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_eps)

            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        elif self.hparams.lr_scheduler != "none":
            assert False, f"Unknown {self.hparams.lr_scheduler=}"

        return [optimizer], [scheduler]



# choosing the config to apply to our model
segformer_config = SegformerConfig(num_channels=6,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2],
                sr_ratios=[1, 1, 1, 1],
                hidden_sizes=[6, 32, 64, 128],
                patch_sizes=[1, 7, 3, 3],
                strides=[1, 4, 2, 2],
                num_attention_heads=[2, 4, 8, 8],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout_prob=0.0,
                initializer_range=0.02,
                drop_path_rate=0.0,
                layer_norm_eps=1e-6,
                decoder_hidden_size=128,
                num_labels=5)



# sending our model into pytorch lightning for training and evaluation
model = pl_IS_model(IsSegformer, segformer_config)



# create our trainer that will handle training
logger = CSVLogger(save_dir=output_path, name="logs")
tb_logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=output_path, name="tb_logs")
callbacks = [ModelCheckpoint(dirpath=output_path, save_weights_only=True,
                             mode="min", monitor="val_loss",
                             save_last=True),
             EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"),
             ]

trainer_args = {'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                'devices': 'auto',
                'num_nodes': 1,
                'strategy': 'auto'}

trainer = pl.Trainer(
    default_root_dir=output_path,
    **trainer_args,
    max_epochs=200,
    log_every_n_steps=1,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=[logger,tb_logger]
)

# fit model
trainer.fit(model, train_loader, validation_loader)



# %%
"""=============================================================================
Validation metrics
============================================================================="""

# if we specified a model checkpoint => reload this specific checkpoint
model = pl_IS_model.load_from_checkpoint(r"C:\Users\afors\Documents\Projects\SPE_IT\output\epoch=99-step=4500.ckpt")

# predict on validation data
validation_outputs = trainer.predict(model, dataloaders=validation_loader)
# note: in pytorch, the output is a list of N elements, N being the number of batches => so we have to convert that to a np array
validation_preds = torch.cat(validation_outputs).detach().cpu().numpy()


export_metrics = dict()

# some general metrics : point precision and IOU
threshold = .5
points = np.arange(1, 304 + 1, 1)
pr = np.zeros((validation_preds.shape[0], 5))
iou = np.zeros((validation_preds.shape[0], 5))
for ix in range(validation_preds.shape[0]):
    for dim in range(5):
        gt = if_y_test[ix, :, dim]
        pd_ = (validation_preds[ix, dim, :] > threshold) * 1
        u = np.sum(gt + pd_ > 0)
        i = np.sum(gt + pd_ == 2)
        if np.isfinite(u):
            iou[ix, dim] = i / u
        else:
            iou[ix, dim] = np.nan
        pr[ix, dim] = np.sum(gt == pd_) / 304


for k in range(iou.shape[1]):
    print("Mean IoU for fraction '{}': {:.2f} +- {:.2f}".format(['G', 'A', 'M', 'k', 'l'][k], np.nanmean(iou[:, k]),
                                                                np.nanstd(iou[:, k])))
    export_metrics['IoU-{}'.format(['G', 'A', 'M', 'k', 'l'][k])] = np.nanmean(iou[:, k])
export_metrics['IoU-global'] = np.nanmean(iou)

for k in range(pr.shape[1]):
    print("Mean accuracy for fraction '{}': {:.2f} +- {:.2f}".format(['G', 'A', 'M', 'k', 'l'][k], np.nanmean(pr[:, k]),
                                                                     np.nanstd(pr[:, k])))


# a function for plotting
def plotITPredictions(ix):
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    # on récupère la class map (binarisée)
    # class_map = y[ix].max(axis=1)
    curve_values = if_x_test[ix, :, :]
    for num, col, lab in zip(range(6), ['black', 'purple', 'pink', 'green', 'red', 'blue'],
                             ['Ref', 'G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304), curve_values[:, num], '-', color=col, label=lab)
    plt.title('Valid set curve index {}'.format(ix))

    plt.legend()
    # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
    #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

    # on plot aussi les autres courbes
    plt.subplot(3, 1, 2)
    for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304) + 1, if_y_test[ix, :, num] / 5 + (4 - num) / 5, '-', color=col, label=lab)
    plt.ylim(-.05, 1.05)
    plt.legend()
    plt.title('Ground truth maps')

    plt.subplot(3, 1, 3)
    for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
        plt.plot(np.arange(0, 304) + 1, validation_preds[ix, num, :] / 5 + (4 - num) / 5, '-', color=col, label=lab)
    plt.ylim(-.05, 1.05)
    plt.legend()
    plt.title('Predicted maps')
    plt.show()

plotITPredictions(1)




# Calculons pour chaque pic réel/prédit la concordance
threshold = 0.5  # ou 0.5
curve_ids = []
groundtruth_spikes = []
predicted_spikes = []
for ix in range(if_x_test.shape[0]):
    flat_gt = np.zeros_like(if_y_test[ix, :, 0])
    for i in range(if_y_test.shape[-1]):
        flat_gt += if_y_test[ix, :, i] * (1 + np.power(2, i))
    gt_starts = []
    gt_ends = []
    prev_v = 0
    for i in range(304):
        if flat_gt[i] != prev_v:  # changed
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

    if len(gt_starts) > 0:
        # pour chaque pic, on détecte ce que le modèle a rendu a cet endroit comme type d'Ig
        for pstart, pend in zip(gt_starts, gt_ends):
            gt_ig_denom = ''
            if np.sum(if_y_test[ix, pstart:pend, :3]) > 0:
                HC_gt = int(np.median(np.argmax(if_y_test[ix, pstart:pend, :3], axis=1)))
                gt_ig_denom = ['G', 'A', 'M'][HC_gt]
            lC_gt = int(np.median(np.argmax(if_y_test[ix, pstart:pend, 3:], axis=1)))
            gt_ig_denom += ['k', 'l'][lC_gt]

            pred_ig_denom = ''
            if np.sum(validation_preds[ix, :, pstart:pend] > threshold) > 0:  # un pic a été détecté
                if np.sum(validation_preds[ix,  :3, pstart:pend] > threshold) > 0:
                    HC_pred = int(np.median(np.argmax(validation_preds[ix,  :3, pstart:pend], axis=0)))
                    pred_ig_denom = ['G', 'A', 'M'][HC_pred]
                lC_pred = int(np.median(np.argmax(validation_preds[ix, 3:, pstart:pend], axis=0)))
                pred_ig_denom += ['k', 'l'][lC_pred]
            else:
                pred_ig_denom = 'none'

            groundtruth_spikes.append(gt_ig_denom)
            predicted_spikes.append(pred_ig_denom)
            curve_ids.append(ix)
    else:
        gt_ig_denom = 'none'
        pred_ig_denom = ''
        if np.sum(validation_preds[ix, :3, :] > threshold) > 0 :
            HC_pred = int(np.median(np.argmax(validation_preds[ix, :3, :], axis=0)))
            pred_ig_denom = ['G', 'A', 'M'][HC_pred]
        lC_pred = int(np.median(np.argmax(validation_preds[ix, 3:, :], axis=0)))
        pred_ig_denom += ['k', 'l'][lC_pred]

        groundtruth_spikes.append(gt_ig_denom)
        predicted_spikes.append(pred_ig_denom)
        curve_ids.append(ix)

conc_df = pd.DataFrame(dict(ix=curve_ids,
                            true=groundtruth_spikes,
                            pred=predicted_spikes))

print(pd.crosstab(conc_df.true, conc_df.pred))

print('Global precision: ' + str(round(100 * np.sum(conc_df.true == conc_df.pred) / conc_df.shape[0], 1)))
for typ in np.unique(conc_df.true):
    subset = conc_df.true == typ
    print('  Precision for type ' + typ + ': ' + str(
        round(100 * np.sum(conc_df.true.loc[subset] == conc_df.pred.loc[subset]) / np.sum(subset), 1)))
    export_metrics['Acc-{}'.format(typ)] = 100 * np.sum(conc_df.true.loc[subset] == conc_df.pred.loc[subset]) / np.sum(
        subset)
export_metrics['Acc-global'] = 100 * np.sum(conc_df.true == conc_df.pred) / conc_df.shape[0]

export_metrics['Mistakes-total'] = conc_df.loc[conc_df.true != conc_df.pred, :].shape[0]
mistakes = conc_df.loc[conc_df.true != conc_df.pred, 'ix'].unique().tolist()
export_metrics['Mistakes-curves'] = len(mistakes)






