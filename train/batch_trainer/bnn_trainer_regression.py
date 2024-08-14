import jax.numpy as np
from jax import random
from torch.utils import data
import matplotlib.pyplot as plt
from optimiser.enkf_bnn import Ensemble_BNN
from jax.nn import relu, tanh


class DataGenerator_batch(data.Dataset):
    def __init__(self, x, y, batch_size=64, rng_key=random.PRNGKey(1)):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.key = rng_key
        self.n_data, self.n_dim = x.shape

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        idx = random.choice(subkey, self.n_data, (self.batch_size,), replace=False)
        return self.x[idx, :], self.y[idx, :]
    

def train_and_evaluate():
    key_data, key_dataset, key_model, key_train, key_noise = random.split(random.PRNGKey(10), 5)
    n_samples = 5000
    batch_size = 64
    N_ensemble = 100
    n_iter = 500
    std_params = 0.02
    std_data = 0.1

    x = random.normal(key_data, (n_samples, 1))
    y = np.sin(np.pi * x) + std_data * random.normal(key_noise, x.shape)

    dataset = DataGenerator_batch(x, y, batch_size=batch_size, rng_key=key_dataset)
    network_layers = [1, 20, 20, 1]
    activation = tanh
    bnn = Ensemble_BNN(network_layers, N_ensemble, activation=activation, rng_key=key_model)

    bnn.eki_train(key_train, dataset, n_iter=n_iter, std_params=std_params, std_data=std_data)

    plt.loglog(bnn.loss_log)
    plt.title('Training Average Data Misfit over Iteraration')
    plt.xlabel('EKI Iteration')
    plt.ylabel('Data Misfit')
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()