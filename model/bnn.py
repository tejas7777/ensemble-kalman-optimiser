import jax.numpy as np
from jax import random
from jax.nn import relu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn

def mlp(layers, activation=relu):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b
        keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            inputs = activation(np.dot(inputs, W) + b)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply


class BNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation=nn.ReLU, prior_mu=0.0, prior_sigma=1.0):
        super(BNN, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h_dim in hidden_layers:
            self.layers.append(bnn.BayesLinear(prior_mu, prior_sigma, in_features=in_dim, out_features=h_dim))
            self.layers.append(activation())
            in_dim = h_dim
        self.layers.append(bnn.BayesLinear(prior_mu, prior_sigma, in_features=in_dim, out_features=output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class BNNClassification(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation=nn.ReLU):
        super(BNNClassification, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.layers.append(activation())
            in_dim = h_dim
        self.layers.append(nn.Linear(in_dim, output_size))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax(x)
        return x

