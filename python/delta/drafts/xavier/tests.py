# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:21:19 2021

@author: afors
"""

import re
import itertools

log_path = r"D:\Anaconda datasets\Capillarys\IF_transformer\x_trans1.log"
with open(log_path, "r") as f :
    txt = f.read()
   
n_models = re.findall("Epoch 1/200", txt)

model_dict = dict()

for i in range(len(n_models)) :
    model_start_flag = re.search(r"Epoch 1/200", txt)
    txt = txt[model_start_flag.span()[1]:]    
    model_stop_flag = re.search("rEpoch 1/200", txt)
    model_dict[i] = txt[:model_start_flag.span()[0]]


model_result = dict()

for model_name, _ in model_dict.items() : 
    epochs = re.findall("val_main_out_loss: \d+\.\d+", model_dict[model_name])
    losses = list()
    for i in range(len(epochs)) :
        losses.append(float(epochs[i].lstrip('val_main_out_loss: ')))
    min_loss = min(losses)
    
    epochs = re.findall("val_main_out_curve_iou: \d+\.\d+", model_dict[model_name])
    ious = list()
    for i in range(len(epochs)) :
        ious.append(float(epochs[i].lstrip('val_main_out_curve_iou: ')))
    max_iou = max(ious)
        
    model_result[model_name] = (min_loss, max_iou)
        

num_heads = [1, 2, 3]  # Number of attention heads
embed_dims = [24, 48, 96] # "embedding" size (the 6 curves, or more)
num_blocks = [1, 3, 6, 9, 12] # number of transformer block in each layer
num_filters = [256, 512, 1024] # number of filters of conv decoders
num_convblocks = [1, 2, 3]

hparams_list = [embed_dims, num_blocks, num_heads, num_filters, num_convblocks]
combs= list(itertools.product(*hparams_list))
combs[47]
