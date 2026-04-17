#!/usr/bin/env python3
"""
Exercises from https://github.com/karpathy/micrograd/tree/master
"""

import math
import numpy as np

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # This allows operations with int and floats
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Sum operation propagates the grad of the parent into the children
            self.grad += 1.0 * out.grad # Initially gradients were set with equal, but that created a bug when a variable was used more than
                                        # once. Solution: following multivariate chain rule, we use += to accumulate the gradient.
            other.grad += 1.0 * out.grad 
        out._backward = _backward

        return out
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # See this line's note in __add__
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Multiplication operation is equal to the value of the other times the grad of the parent 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        # a.__mul__(2) != 2.__mul__(a)
        # Adds support for other * self
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # The local derivative of e**x operation is equal to  e**x = out.data
            self.grad += out.data * out.grad 
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # The local derivative of tanh operation is equal to  1 - tanh(n)**2 = 1 - o**2
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self, verbose=True):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                if verbose:
                    print(f'visiting: {v.label}')
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                if verbose:
                    print(f'Appending {v.label} to topo')
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            if verbose:
                print(f'Calling _backward in {v.label}')
            v._backward()