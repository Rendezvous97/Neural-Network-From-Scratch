import matplotlib.pyplot as plt
import random
from driver import Value


# class Module is the way pyTorch implements their neural network class as well

class Module:

    def zero_grad(self): # To make all gradients 0 before back propagating during each optimization
        for p in self.parameters():
            p.grad = 0

    def parameters(self): # This is overriden by the parameters function of each child class—Neuron, Layer, MLP
        return []


# Each class has an __init__ function, __call__ function and parameters function

class Neuron(Module): # is a sub class of class Module

    def __init__(self, nin): #nin in this case is the number of input features (not the samples) and for each of the next layers, it's the number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):

    def __init__(self, nin, nout): # nout is the number of neurons in a single layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP(Module):

    def __init__(self, nin, nouts): #nouts will be a list of hidden and final layers like [3, 3, 1]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

