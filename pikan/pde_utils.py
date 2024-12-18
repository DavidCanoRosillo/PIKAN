import jax
import jax.numpy as jnp
from pikan.model_utils import gradf
import optax

# from https://arxiv.org/pdf/2407.17611
def get_pde_Helmholtz(model, modeltype="MLP"):
    @jax.jit
    def pde_loss_fn(params, collocs, state):
        def u(vec_x):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        if modeltype == "KAN":
            variables = {'params' : params, 'state': state}
            
        y = u(collocs)
        u_xx = gradf(u, 0, 2)(collocs)
        u_yy = gradf(u, 1, 2)(collocs)
        
        f = (1-17*jnp.pi**2)*jnp.sin(jnp.pi*collocs[:, 0])*jnp.sin(4*jnp.pi*collocs[:, 1])
        f = f.reshape(-1,1)
        
        pde_residual = u_xx + u_yy + y - f
        
        return pde_residual

    return pde_loss_fn

# from https://arxiv.org/pdf/2407.17611
def get_pde_heat1(model, modeltype="MLP"):
    @jax.jit 
    def pde_loss_fn(params, collocs, state):
        def u(vec_x):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        if modeltype == "KAN":
            variables = {'params' : params, 'state': state}
            
        y = u(collocs)
        u_xx = gradf(u, 0, 2)(collocs)
        u_t = gradf(u, 1, 1)(collocs)

        f = (jnp.pi**2-1)*jnp.exp(-collocs[:, 1])*jnp.sin(jnp.pi*collocs[:, 0])
        f = f.reshape(-1,1)
        
        pde_residual = u_t - u_xx - f
        
        return pde_residual

    return pde_loss_fn

# from https://arxiv.org/pdf/2407.17611
def get_pde_burgers1(model, modeltype="MLP"):
    @jax.jit 
    def pde_loss_fn(params, collocs, state):
        def u(vec_x):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        if modeltype == "KAN":
            variables = {'params' : params, 'state': state}
            
        y = u(collocs)
        u_xx = gradf(u, 0, 2)(collocs)
        u_x = gradf(u, 0, 1)(collocs)
        u_t = gradf(u, 1, 1)(collocs)
        
        v = 0.01 / jnp.pi
        pde_residual = u_t + y*u_x - v*u_xx
        
        return pde_residual

    return pde_loss_fn

from jax.lax import stop_gradient

def get_adaptive_loss(model, pde_loss_fn, modeltype="MLP"):
    @jax.jit
    def adaptive_loss(params, collocs, bc_collocs, bc_data, state, loc_w):
        def u(vec_x):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        if modeltype == "KAN":
            variables = {'params' : params, 'state': state}
        
        new_loc_w = []

        eta = jnp.array(0.0001, dtype=float)

        pde_residues = pde_loss_fn(params, collocs, state)
        abs_res = jnp.abs(pde_residues)        
        loc_w_pde = ((jnp.array(1.0)-eta)*loc_w[0]) + ((eta*abs_res)/jnp.max(abs_res))

        # apply the operation described in these three lines, so called temporal weights
        
        # cum_loss = jnp.cumsum(stop_gradient(pde_residues))
        # temporal_weights = jnp.exp(-cum_loss / jnp.max(cum_loss))
        
        # pde_loss = jnp.mean((pde_residues*temporal_weights*loc_w_pde)**2)
        
        pde_loss = jnp.mean((pde_residues*loc_w_pde)**2)
        new_loc_w.append(loc_w_pde)
        
        bc_loss = 0
        for idx, bc_colloc in enumerate(bc_collocs):
            bc_residues = u(bc_colloc) - bc_data[idx]
            abs_res = jnp.abs(bc_residues)
            loc_w_bc = ((jnp.array(1.0)-eta)*loc_w[idx+1]) + ((eta*abs_res)/jnp.max(abs_res))
           
            bc_loss += jnp.mean((bc_residues*loc_w_bc)**2) 
            new_loc_w.append(loc_w_bc)
            
        loss = bc_loss + 0.01*pde_loss
        # loss = bc_loss + pde_loss
        
        return loss, new_loc_w

    return adaptive_loss

def get_vanilla_loss(model, pde_loss_fn):
    @jax.jit
    def vanilla_loss(params, collocs, bc_collocs, bc_data, state, new_loc_w):
        def u(vec_x):
            y = model.apply(variables, vec_x)
            return y
        variables = {'params' : params}
        
        pde_residual = pde_loss_fn(params, collocs, state)
        
        bc_loss = 0
        for (bc_x, bc_y) in zip(bc_collocs, bc_data):
            bc_loss += jnp.mean((u(bc_x) - bc_y)**2)
        
        loss = jnp.mean(pde_residual**2) + bc_loss
        return loss, new_loc_w

    return vanilla_loss

def get_pde_train_step(model, optimizer, loss_fn):
    @jax.jit
    def train_step(params, collocs, bc_collocs, bc_data, opt_state, state, loc_w):
        # Compute gradients and loss
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, new_loc_w), grads = grad_fn(params, collocs, bc_collocs, bc_data, state, loc_w)

        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, new_loc_w

    return train_step

if __name__ == "__main__":
    print("Correctly installed?")