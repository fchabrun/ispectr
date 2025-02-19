import math
from collections import OrderedDict
from dataclasses import fields, dataclass
from typing import Tuple, Any, Optional, Union

import lightning as pl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, trunc_normal_


"""
========================================================================================================================
########## SEGFORMER FOR 1D SPE IS DATA
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/segformer/
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""



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
            depths=(2, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
            hidden_sizes=(32, 64, 160, 256),
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            num_attention_heads=(1, 2, 5, 8),
            mlp_ratios=(4, 4, 4, 4),
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=256,
            semantic_loss_ignore_index=255,
            num_labels=0,
            use_return_dict=True,
            output_hidden_states=True,
            output_attentions=False
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
            iterator = None
            if isinstance(first_field, dict):
                iterator = first_field.items()
            else:
                try:
                    iterator = iter(first_field)
                except TypeError:
                    pass

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if iterator is not None:
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
        # _, _, length = embeddings.shape
        # base impl : (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
        # but we already have a sequence without height and width so we don't need to apply  flatten(2) to embeddings
        # this can be fed to a Transformer layer
        embeddings = embeddings.transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, length


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio=1):
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
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio=1):
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

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, mlp_ratio, sequence_reduction_ratio=1):
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

        # batch_size = pixel_values.shape[0]

        hidden_states = pixel_values

        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, length = embedding_layer(hidden_states)
            # print(length)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, length, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            # if idx != len(self.patch_embeddings) - 1 or (
            #        idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            # ):
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
            # if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
            #    height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
            #    encoder_hidden_state = (
            #        encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            #    )

            # unify channel dimension
            length = encoder_hidden_state.shape[1]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            # encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, length)
            # upsample
            # print(encoder_hidden_state.shape)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[1], mode="linear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
            # print([all_hidden_states[i].shape for i in range(len(all_hidden_states))])

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        # print(hidden_states.shape)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # print(hidden_states.shape)
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

        pixel_values = pixel_values.squeeze(dim=2).transpose(1, 2)
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # print([outputs.hidden_states[i].shape for i in range(len(outputs.hidden_states))])
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size => not needed here
            # upsampled_logits = nn.functional.interpolate(
            #    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            # )
            if self.config.num_labels > 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.transpose(1, 2))
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



"""
========================================================================================================================
########## SWIN-UNET FOR 1D SPE IS DATA
Adapted from https://github.com/HuCaoFighting/Swin-Unet/tree/main
"""

class SwinUnetConfig:
    r"""
    This is the configuration class to store the configuration of a [`SwinUnetModel`]. It is used to instantiate an
    SwinUnet model according to the specified arguments, defining the model architecture.

    Args:
        # TODO
    ```"""

    model_type = "swinunet"

    def __init__(
            self,
            spe_size=304,
            patch_size=4,
            in_chans=6,
            num_classes=5,
            embed_dim=96,
            depths=[2, 2, 2],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=19,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True

    ):
        super().__init__()

        self.spe_size = spe_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.num_classes = num_classes
        self.ape = ape
        self.patch_norm = patch_norm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of SPE trace

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The length of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * 1 - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(1)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += 1 - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            1 * self.window_size, 1 * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            L = self.input_resolution
            img_mask = torch.zeros((1, L, 1))  # 1 H W 1
            l_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for l in l_slices:
                img_mask[:, l, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, L, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # B L C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
        x = x.view(B, L, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.reduction = nn.Linear(2 * dim, dim, bias=False) # no need here as patch merging only 2*C for 1d data
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        B, L, C = x.shape

        x = x.view(B, L, C)

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        #x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        self.reduction = nn.Linear(dim, dim//2, bias=False)

    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, L, C)
        x = rearrange(x, "b l (p1 c)-> b (l p1) c", p1=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.reduction(x)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, L, C)
        x = rearrange(x, "b l (p1 c)-> b (l p1) c", p1=self.dim_scale,
                      c=C // (self.dim_scale))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            #print("block shape : ", x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
            #print("downsample shape : ", x.shape)
        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            #print("block shape : ", x.shape)
        if self.upsample is not None:
            x = self.upsample(x)
            #print("upsample shape : ", x.shape)
        return x


class PatchEmbed(nn.Module):
    r""" SPE traces to Patch Embedding

    Args:
        spe_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, spe_size=304, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None):
        super().__init__()
        patches_resolution = spe_size // patch_size
        self.spe_size = spe_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.spe_size, \
            f"Input image size {L} doesn't match model ({self.img_size})."
        x = self.proj(x).transpose(1, 2)  # B L C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerForIS(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        spe_size (int): Input image size. Default 304
        patch_size (int): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 6
        num_classes (int): Number of classes for classification head. Default: 5
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 19
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, config,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                config.depths,
                config.depths_decoder, config.drop_path_rate, config.num_classes))

        self.config = config
        self.num_classes = config.num_classes
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.ape = config.ape
        self.patch_norm = config.patch_norm
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(config.embed_dim * 2)
        self.mlp_ratio = config.mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            spe_size=config.spe_size, patch_size=config.patch_size, in_chans=config.in_chans, embed_dim=config.embed_dim,
            norm_layer=config.norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(config.embed_dim * 2 ** i_layer),
                               input_resolution= patches_resolution // (2 ** i_layer),
                               depth=config.depths[i_layer],
                               num_heads=config.num_heads[i_layer],
                               window_size=config.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                               drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                               drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                               norm_layer=config.norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(config.embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    dim=int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=config.norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=patches_resolution // (2 ** (self.num_layers - 1 - i_layer)),
                                         depth=config.depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=config.num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=config.window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                         drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                                         drop_path=dpr[sum(config.depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             config.depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=config.norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = config.norm_layer(self.num_features)
        self.norm_up = config.norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            #print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=4, dim=config.embed_dim)
            self.output = nn.Conv1d(in_channels=config.embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        #print("embed shape : ",x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            #print("encoder pre layer :", x.shape)
            x_downsample.append(x)
            x = layer(x)
            #print("encoder post layer :", x.shape)
        x = self.norm(x)  # B L C
        #print("post encodr :", x.shape)
        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                #print("pre layer 0 up :", x.shape)
                x = layer_up(x)
                #print("post layer 0 up :", x.shape)
            else:
                #print("pre layer up :", x.shape)
                x = torch.cat([x, x_downsample[2 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                #print("post layer 1 up :", x.shape)

        x = self.norm_up(x)  # B L C
        #print("final layer 1 up :", x.shape)
        return x

    def up_x4(self, x):
        B, L, C = x.shape

        if self.final_upsample == "expand_first":
            #print("final pre up :", x.shape)
            x = self.up(x)
            x = x.view(B, 4*L, -1)
            x = x.permute(0, 2, 1)  # B,C,L
            x = self.output(x)
            #print("final post up :", x.shape)

        return x

    def forward(self, x: torch.FloatTensor,
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

        x = x.squeeze(2)
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        logits = self.up_x4(x)

        loss = None
        if labels is not None:
            if self.config.num_classes > 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.transpose(1, 2))

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states= None, # not implemented yet
            attentions=None, # not implemented yet
        )



"""
========================================================================================================================
################# MedNeXt model adapted for 1d data
"""


class MedNeXtConfig:
    r"""
    This is the configuration class to store the configuration of a [`MedNeXtModel`]. It is used to instantiate an
    MedNext model according to the specified arguments, defining the model architecture.

    Args:

    ```"""

    model_type = "mednext"

    def __init__(
            self,
            in_channels: int,
            n_channels: int,
            n_classes: int,
            exp_r: int = 4,  # Expansion ratio as in Swin Transformers
            kernel_size: int = 7,  # Ofcourse can test kernel_size
            enc_kernel_size: int = None,
            dec_kernel_size: int = None,
            deep_supervision: bool = False,  # Can be used to test deep supervision
            do_res: bool = False,  # Can be used to individually test residual connection
            do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
            norm_type='group',
            grn=False,
            neg_slope=1e-2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_channels = n_channels
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.exp_r = exp_r
        self.kernel_size = kernel_size
        self.enc_kernel_size = enc_kernel_size
        self.dec_kernel_size = dec_kernel_size
        self.deep_supervision = deep_supervision
        self.do_res = do_res
        self.do_res_up_down = do_res_up_down
        self.block_counts = block_counts
        self.norm_type = norm_type
        self.grn = grn
        self.neg_slope = neg_slope



class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        # First Conv1D layer with DepthWise Convolutions
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = MedNeXtLayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        # Second convolution (Expansion) layer with Conv1D
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer
        self.conv3 = nn.Conv1d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        #self.grn = grn
        #if grn:
        #    if dim == '3d':
        #        self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
        #        self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
        #    elif dim == '2d':
        #        self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
        #        self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        #if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
        #    if self.dim == '3d':
        #        gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
        #    elif self.dim == '2d':
        #        gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
        #    nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        #    x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type,
                         grn=grn)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type,
                         grn=grn)

        self.resample_do_res = do_res

        if do_res:
            self.res_conv = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        x1 = nn.functional.pad(x1, (1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            res = nn.functional.pad(res, (1, 0))
            x1 = x1 + res

        return x1


class MedNeXtOutBlock(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv_out = nn.Conv1d(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class MedNeXtLayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MedNeXtForIS(nn.Module):

    def __init__(self,
                 config,
                 ):

        super().__init__()

        self.config = config
        self.do_ds = config.deep_supervision

        if config.kernel_size is not None:
            enc_kernel_size = config.kernel_size
            dec_kernel_size = config.kernel_size
        else :
            enc_kernel_size = config.enc_kernel_size
            dec_kernel_size = config.dec_kernel_size

        self.stem = nn.Conv1d(config.in_channels, config.n_channels, kernel_size=1) # padding='same'

        if type(config.exp_r) == int:
            exp_r = [config.exp_r for _ in range(len(config.block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels,
                out_channels=config.n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[0])]
                                         )

        self.down_0 = MedNeXtDownBlock(
            in_channels=config.n_channels,
            out_channels=2 * config.n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 2,
                out_channels=config.n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[1])]
                                         )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * config.n_channels,
            out_channels=4 * config.n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 4,
                out_channels=config.n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * config.n_channels,
            out_channels=8 * config.n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 8,
                out_channels=config.n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[3])]
                                         )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * config.n_channels,
            out_channels=16 * config.n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 16,
                out_channels=config.n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * config.n_channels,
            out_channels=8 * config.n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 8,
                out_channels=config.n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * config.n_channels,
            out_channels=4 * config.n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 4,
                out_channels=config.n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[6])]
                                         )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * config.n_channels,
            out_channels=2 * config.n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels * 2,
                out_channels=config.n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for i in range(config.block_counts[7])]
                                         )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * config.n_channels,
            out_channels=config.n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=config.do_res_up_down,
            norm_type=config.norm_type,
            grn=config.grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=config.n_channels,
                out_channels=config.n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=config.do_res,
                norm_type=config.norm_type,
                grn=config.grn
            )
            for _ in range(config.block_counts[8])]
                                         )

        self.out_0 = MedNeXtOutBlock(in_channels=config.n_channels, n_classes=config.n_classes)

        if config.deep_supervision:
            self.out_1 = MedNeXtOutBlock(in_channels=config.n_channels * 2, n_classes=config.n_classes)
            self.out_2 = MedNeXtOutBlock(in_channels=config.n_channels * 4, n_classes=config.n_classes)
            self.out_3 = MedNeXtOutBlock(in_channels=config.n_channels * 8, n_classes=config.n_classes)
            self.out_4 = MedNeXtOutBlock(in_channels=config.n_channels * 16, n_classes=config.n_classes)

        self.block_counts = config.block_counts

        self.neg_slope = config.neg_slope

        self.apply(self._init_weights)


#    def _init_weights(self, m):
#        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) :
#            m.weight = nn.init.kaiming_normal_(m.weight, a=self.neg_slope)
#            if m.bias is not None:
#                m.bias = nn.init.constant_(m.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) :
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x: torch.FloatTensor,
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

        x = x.squeeze(2)
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            logits = [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            logits =  x

        loss = None
        if labels is not None:
            if self.config.n_classes > 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.transpose(1, 2))

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # not implemented yet
            attentions=None,  # not implemented yet
        )



"""
========================================================================================================================
################# Generic pytorch Lightning class to wrap and train models with
"""

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
                 adam_eps=1e-08,
                 weight_decay=0.0
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
        self.log("train_loss", outputs.loss)
        # print([outputs.logits[i].shape for i in range(4)], [outputs.loss[i].shape for i in range(4)], [outputs.hidden_states[i].shape for i in range(4)])
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_outputs = self.model(x)

        # print(y.shape, val_outputs.logits.shape)
        self.y_true.append(y.transpose(1, 2))
        self.y_pred.append(val_outputs.logits)
        # print([self.y_true[i].shape for i in range(len(self.y_true))], [self.y_pred[i].shape for i in range(len(self.y_pred))])

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        # print(y_true.shape, y_pred.shape)
        val_loss = nn.BCEWithLogitsLoss()
        ce_loss = val_loss(y_pred, y_true)
        # accuracy = torch.sum(torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)) / y_true.shape[0]

        log_dict = {'val_loss': ce_loss}
        # for i in range(self.n_classes):
        #    preds = torch.argmax(y_pred[torch.argmax(y_true, dim=1) == i], dim=1)
        #    log_dict[f"val_accuracy_class_{i}"] = torch.sum(preds == i) / preds.shape[0]

        self.log_dict(log_dict)

        self.y_true.clear()
        self.y_pred.clear()

    def configure_optimizers(self):
        optimizer = None
        scheduler = None

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


