import jax
import jax.numpy as jnp

# Function to test interpolation
@jax.jit
def oscillatory_sine(x, FREQ):
    y = jnp.sin(FREQ * jnp.pi * x[:, 0]) * jnp.cos(FREQ * jnp.pi * x[:, 1])
    return y

@jax.jit
def radial_oscillation(x, FREQ):
    r = jnp.sqrt((x[:, 0] - 1)**2 + (x[:, 1] - 1)**2)
    y = jnp.sin(FREQ * jnp.pi * r)
    return y

@jax.jit
def discontinuous_starburst(x, FREQ):
    theta = jnp.arctan2(x[:, 1] - 1, x[:, 0] - 1)
    y = jnp.where(jnp.mod(theta * FREQ, 2 * jnp.pi) > jnp.pi, 1.0, -1.0)
    return y

@jax.jit
def ring_function(x, FREQ):
    y = jnp.sin(FREQ * jnp.pi * jnp.sqrt((x[:, 0] - 1)**2 + (x[:, 1] - 1)**2))
    return y

@jax.jit
def modulated_gaussian(x, FREQ):
    y = jnp.exp(-((x[:, 0] - 1)**2 + (x[:, 1] - 1)**2)) * \
        jnp.sin(FREQ * jnp.pi * x[:, 0]) * jnp.cos(FREQ * jnp.pi * x[:, 1])
    return y

@jax.jit
def circular_wave_interference(x, FREQ):
    r1 = jnp.sqrt((x[:, 0] - 0.5)**2 + (x[:, 1] - 0.5)**2)
    r2 = jnp.sqrt((x[:, 0] - 1.5)**2 + (x[:, 1] - 1.5)**2)
    y = jnp.sin(FREQ * jnp.pi * r1) + jnp.sin(FREQ * jnp.pi * r2)
    return y