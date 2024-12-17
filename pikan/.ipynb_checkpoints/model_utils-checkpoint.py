# model_utils.py

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import Callable, Optional
from dataclasses import field
from typing import List
from scipy.stats.qmc import Sobol
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxkan.models.KAN import KAN

class FourierFeats(nn.Module):
    num_output: int
    
    std = 10
    mean = 0
    
    @nn.compact
    def __call__(self, x):
        B = self.param(
            'B', lambda rng, shape: self.mean + jax.random.normal(rng, shape) * self.std,
           (x.shape[-1], self.num_output // 2)
        )
        bias = self.param(
            'bias', lambda rng, shape: jax.random.normal(rng, shape) * 0,
            (self.num_output // 2,)
        )
        
        x = jnp.matmul(x, B)
        x = jnp.concatenate([jnp.cos((x + bias)), jnp.sin((x + bias))], axis=-1)
            
        return x
    
class FourierKAN(nn.Module):
    kernel_init: Callable
    num_input: int
    num_output: int
    use_fourier_feats: bool = False  # Whether to use FourierFeats as the first layer
    layer_sizes: List[int] = field(default_factory=list)
    
    @nn.compact
    def __call__(self, x):
        if self.use_fourier_feats:
            x = FourierFeats(self.layer_sizes[0])(x)
            layer_dims = list(self.layer_sizes) + [self.num_output]
        else:
            layer_dims = [self.num_input] + list(self.layer_sizes) + [self.num_output]

        model = KAN(
            layer_dims=layer_dims,
            k=3, 
            add_bias=True
        )
        y, spl_regs = model(x)
        
        return y

class GeneralizedMLP(nn.Module):
    kernel_init: Callable
    num_input: int
    num_output: int
    use_fourier_feats: bool = False  # Whether to use FourierFeats as the first layer
    layer_sizes: List[int] = field(default_factory=list)

    std = 10
    mean = 0

    @nn.compact
    def __call__(self, x):
        # Add hidden layers
        for idx, size in enumerate(self.layer_sizes):
            if self.use_fourier_feats and idx==0:
                x = FourierFeats(size)(x)
            else:
                x = nn.Dense(size, kernel_init=self.kernel_init)(x)
                x = nn.tanh(x)
            
        # Final output layer
        x = nn.Dense(self.num_output, kernel_init=self.kernel_init)(x)
        return x

class PirateBlock(nn.Module):
    kernel_init: Callable
    num_hidden: int
    
    @nn.compact
    def __call__(self, x, U, V):
        eye = x

        x = nn.tanh(nn.Dense(self.num_hidden, kernel_init=self.kernel_init)(x))  # f = tanh(Dense(x))
        x = x * U + (1 - x) * V  # z_1
        
        x = nn.tanh(nn.Dense(self.num_hidden, kernel_init=self.kernel_init)(x))  # g = tanh(Dense(z_1))
        x = x * U + (1 - x) * V  # z_2
        
        x = nn.tanh(nn.Dense(self.num_hidden, kernel_init=self.kernel_init)(x))  # h = tanh(Dense(z_2))
    
        alpha = self.param('alpha', lambda rng: 0.)
        
        return alpha * x + (1-alpha) * eye 
    
class PirateNet(nn.Module):
    kernel_init: Callable
    num_input: int
    num_output: int
    layer_sizes: List[int] = field(default_factory=list)
    init_last_W: Optional = None  # This is the parameter that defaults to None
    
    @nn.compact
    def __call__(self, x):
        # Add hidden layers
        for idx, size in enumerate(self.layer_sizes):
            if idx==0:
                x = FourierFeats(size)(x)
                
                U = nn.Dense(size, kernel_init=self.kernel_init)(x)
                U = nn.tanh(U)

                V = nn.Dense(size, kernel_init=self.kernel_init)(x)
                V = nn.tanh(V)
            else:
                x = PirateBlock(self.kernel_init, size)(x, U, V)
            
        # Final output layer
        x = nn.Dense(self.num_output, kernel_init=self.kernel_init)(x)
        return x
    
class KeyHandler:
    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        
    def key(self):
        _, self._key = jax.random.split(self._key)
        return self._key

def gradf(f, idx, order=1):
    '''
        Computes gradients of arbitrary order
        
        Args:
        -----
            f (function): function to be differentiated
            idx (int): index of coordinate to differentiate
            order (int): gradient order
        
        Returns:
        --------
            g (function): gradient of f
    '''
    def grad_fn(g, idx):
        return lambda tx: jax.grad(lambda tx: jnp.sum(g(tx)))(tx)[..., idx].reshape(-1,1)

    g = lambda tx: f(tx)
    
    for _ in range(order):
        g = grad_fn(g, idx)
        
    return g

def sobol_sample(X0, X1, N, seed=None):
    '''
        Performs Sobol sampling
        
        Args:
        -----
            X0 (np.ndarray): lower end of sampling region
                shape (dims,)
            X1 (np.ndarray): upper end of sampling region
                shape (dims,)
            N (int): number of points to sample
            seed (int): seed for reproducibility
        
        Returns:
        --------
            points (np.ndarray): sampled points
                shape (N,dims)
    '''
    dims = X0.shape[0]

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)  # Ensure seed fits in 32 bits
    
    sobol_sampler = Sobol(dims, scramble=True, seed=seed)
    points = sobol_sampler.random_base2(int(np.log2(N)))
    points = X0 + points * (X1 - X0)
    
    return points

def get_mse_loss(model, MODEL='MLP'):
    @jax.jit
    def mse_loss_mlp(params, x, y, state, loc_w):
        def u(vec_x, variables):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        
        y_hat = u(x, variables)
        loss = jnp.mean((y_hat - y)**2)

        new_loc_w = loc_w
        return loss, new_loc_w

    if MODEL == 'MLP' or MODEL == 'PIRATE':
        return mse_loss_mlp
    
    @jax.jit
    def mse_loss_kan(params, x, y, state, loc_w):
        def u(vec_x, variables):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params, 'state': state}
        
        y_hat = u(x, variables)
        loss = jnp.mean((y_hat - y)**2)

        new_loc_w = loc_w
        return loss, new_loc_w
    
    if MODEL == 'KAN':
        return mse_loss_kan
    
def get_train_step(model, optimizer, loss_fn):
    @jax.jit
    def train_step(params, x, y, opt_state, state, loc_w):
        # Compute gradients and loss
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, new_loc_w), grads = grad_fn(params, x, y, state, loc_w)

        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, new_loc_w

    return train_step
