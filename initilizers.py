#the contents of this file are based on suplementary material to Jumper, John et al. 2021. “Highly Accurate Protein Structure Prediction with AlphaFold.” Nature 596 (7873): 583–89.
#and the initiation scheme presented therein (section 1.11.4)
#and the HAIKU package code https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py

import math

from torch import nn

trunc_normal_correction = 0.87962566103423978   #constant from the above implementation, they take it from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)

def fan_in(layer: nn.Linear):
    return layer.weight.size(1)

def fan_out(layer: nn.Linear):
    return layer.weight.size(0)

def linear_init_with_lecun_uniform(layer: nn.Linear):
    scale = math.sqrt(3 / fan_in(layer))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_lecun_normal(layer: nn.Linear):
    #lecun normal is truncated normal in fact - see https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py
    stdev = math.sqrt(1 / fan_in(layer))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_glorot_uniform(layer: nn.Linear):
    scale = math.sqrt(6 / (fan_in(layer) + fan_out(layer)))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_glorot_normal(layer: nn.Linear):
    return layer
    #glorot normal is truncated normal in fact - see https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py
    stdev = math.sqrt(2 / (fan_in(layer) + fan_out(layer)))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_he_uniform(layer: nn.Linear):
    scale = math.sqrt(6 / fan_in(layer))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_he_normal(layer: nn.Linear):
    #he normal is truncated normal in fact - see https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py
    stdev = math.sqrt(2 / fan_in(layer))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()    #in contrast to what is default in PyTorch, in Alphafold paper they initialize biases with 0
    return layer

def linear_init_with_zeros(layer: nn.Linear):
    layer.weight.data.zero_()
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer

