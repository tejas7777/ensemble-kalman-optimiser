import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from functools import partial
import itertools

from model.bnn import mlp
from jax.tree_util import tree_leaves, tree_unflatten, tree_structure, tree_flatten, tree_map
import jax.numpy as np
from jax import random
from jax.nn import relu
from tqdm import tqdm


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = tree_structure(target)
    keys = random.split(rng_key, treedef.num_leaves)
    return tree_unflatten(treedef, keys)

def tree_random_normal_like(rng_key, target, std):
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(lambda l, k: l + std * random.normal(k, l.shape, l.dtype), target, keys_tree)

class Ensemble_BNN:
    def __init__(self, network_layers, N_ensemble, activation=relu, rng_key=random.PRNGKey(0)):
        key_net, key = random.split(rng_key, 2)
        self.net_init, self.net_apply = mlp(network_layers, activation)
        self.N_ensemble = N_ensemble
        self.v_params = vmap(self.net_init)(random.split(key_net, N_ensemble))
        self.flat_arrays, self.treedef = tree_flatten(self.v_params)
        self.itercount = itertools.count()
        self.loss_log = []

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, v_params):
        flat_arrays, _ = tree_flatten(v_params)
        flat_arrays_list = [np.reshape(array, (self.N_ensemble, -1)) for array in flat_arrays]
        flat_vector = np.concatenate(flat_arrays_list, 1)
        return flat_vector.T

    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, flat_vector):
        start_idx = 0
        reshaped_arrays = []
        flat_vector = flat_vector.T
        for array in self.flat_arrays:
            size = array[0, :].size
            shape = array.shape
            reshaped_arrays.append(np.reshape(flat_vector[:, start_idx: start_idx + size], shape))
            start_idx += size
        new_v_params = tree_unflatten(self.treedef, reshaped_arrays)
        return new_v_params

    @partial(jit, static_argnums=(0,))
    def avg_misfit_loss(self, v_params, batch):
        x, y = batch
        num_data = len(y)
        Y = vmap(self.net_apply, in_axes=(0, None))(v_params, x).squeeze(2).T
        loss = np.mean(np.linalg.norm(y - Y, axis=1) / num_data)
        return loss

    @partial(jit, static_argnums=(0,))
    def eki_step(self, v_params, batch, std_data, key):
        x, y = batch
        num_data = len(y)
        
        R = std_data**2 * np.ones(y.size)
        sqrt_R_vec = np.sqrt(R).reshape((num_data, -1))
        Y = vmap(self.net_apply, in_axes=(0, None))(v_params, x).squeeze(2).T
        d = y - Y + sqrt_R_vec * random.normal(key, Y.shape)
        Y_mean = np.mean(Y, axis=1, keepdims=True)
        YY = (Y - Y_mean) / np.sqrt(self.N_ensemble - 1)
        C_yy = YY @ YY.T

        X = self.flatten_params(v_params)
        X_mean = np.mean(X, axis=1, keepdims=True)
        XX = (X - X_mean) / np.sqrt(self.N_ensemble - 1)
        C_xy = XX @ YY.T

        v_params_flat = X + C_xy @ (np.linalg.solve(C_yy + np.diag(R), d))
        v_params = self.unflatten_params(v_params_flat)
        return v_params

    def eki_train(self, key, dataset, n_iter=10000, std_params=0.002, std_data=0.01):
        key_next, key_X, key_Y = random.split(key, 3)
        data = iter(dataset)
        random_perturb = lambda key, std_params, params: tree_random_normal_like(key, params, std_params)
        v_params = self.v_params

        for it in tqdm(range(n_iter)):
            key_next, key_X, key_Y, key_adaptive = random.split(key_next, 4)
            v_params = random_perturb(key_X, std_params, v_params)
            batch = next(data)
            v_params = self.eki_step(v_params, batch, std_data, key_Y)
            loss = self.avg_misfit_loss(v_params, batch)
            self.loss_log.append(loss)
        self.v_params = v_params

    @partial(jit, static_argnums=(0,))
    def predict(self, x):
        pred_s = vmap(self.net_apply, (0, None))(self.v_params, x)
        return pred_s.squeeze(2).T