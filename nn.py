#!/usr/bin/env python3
"""
Manual implementation of neural network.

Exercises from https://github.com/karpathy/micrograd/tree/master
"""

import random
import math
import numpy as np

from engine import Value

class Neuron:

    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, n_inputs, n_output):
        self.neurons = [Neuron(n_inputs) for _ in range(n_output)] # n_output is equivalent to number of neurons in the layer

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # Can be generated with list comprenhesion but keeping the loop for reference
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
class MLP:

    def __init__(self, n_inputs, n_outputs):
        layer_size = [n_inputs] + n_outputs # n_outputs is a list of the sizes of each layer
        self.layers = [Layer(layer_size[i], layer_size[i+1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]