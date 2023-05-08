import numpy as np


class Module:
    
    def forward(self, *args, **kwargs):
        pass

    
class Network(Module):
    
    def __init__(self, layers=None):
        # store the list of layers passed in the constructor in your Network object
        self.layers = layers
        pass
    
    def forward(self, x):
        # for executing the forward pass, run the forward passes of each
        # layer and pass the output as input to the next layer
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def add_layer(self, layer):
        # append layer at the end of the list of already existing layer
        self.layers.append(layer)
        pass

    
class LinearLayer(Module):
    
    def __init__(self, W, b):
        # store parameters W and b
        self.W = W
        self.b = b
        pass
    
    def forward(self, x):
        # compute the affine linear transformation x -> Wx + b
        return self.W @ x + self.b

    
class Sigmoid(Module):
    
    def forward(self, x):
        # implement the sigmoid 𝑥 to 𝑒^𝑥/(𝑒^𝑥+1)
        self.x = x
        return np.exp(self.x) / (np.exp(self.x) + 1)

    
class ReLU(Module):
    
    def forward(self, x):
        # implement a ReLU 𝑥 is mapped to max(𝑥,0)
        self.x = x
        return np.maximum(self.x, 0)

    
class Loss(Module):
    
    def forward(self, prediction, target):
        pass


class MSE(Loss):
    
    def forward(self, prediction, target):
        # implement MSE loss: mean squared difference of prediction and target.
        self.prediction = prediction
        self.target = target
        return np.mean((self.prediction - self.target) ** 2)


class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        # implement cross entropy loss: ℓ(𝑥,𝑙)=−log(𝜎𝑙(𝑥)) with 𝜎l(x)=𝑒^xl / ∑to𝐿from𝑖=1 𝑒^x𝑖
        self.prediction = prediction
        self.target = target
        return np.negative(
            np.log(
                (np.exp(self.prediction) / np.exp(self.prediction).sum())
                [self.target]
            )
        )
