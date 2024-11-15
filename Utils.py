import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol
import numpy as np
import matplotlib.pyplot as plt
import time

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