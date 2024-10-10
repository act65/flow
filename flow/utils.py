import jax.numpy as jnp
from jax import jacfwd, random
from jax import lax, jit, vmap
from jax.tree_util import tree_map

class FlattenFn():
    def __init__(self, fn, argnum=0):
        self.fn = fn
        self.shape = None
        self.argnum = argnum

    def __call__(self, *args):
        # receives flat inputs. outputs flat outputs
        args = [arg.reshape(self.shape) if i == self.argnum else arg for i, arg in enumerate(args)]
        return self.fn(*args).ravel()
    
    def flatten_in(self, *args):
        self.shape = args[self.argnum].shape
        return [arg.ravel() if i == self.argnum else arg for i, arg in enumerate(args)]
    
    def flatten_out(self, *args):
        return [arg.reshape(self.shape) if i == self.argnum else arg for i, arg in enumerate(args)]

def divergence(fn, argnums=0):
    """
    This could be written more simply. 
    But we wan to support arbitrary shapes (sequences, images, etc).
    So we need to construct a wrapper function that flattens the input and outputs.
    Then we can take the jacobian of the flattened function and calculate its div.
    """
    flat_fn = FlattenFn(fn, argnum=argnums)
    jac_fn = jacfwd(flat_fn, argnums=argnums)
    def _divergence(*args):
        # https://github.com/google/jax/issues/3022#issuecomment-2100553108
        # TODO more effieient to just calculate the df_i/dx_is separately?
        flat_args = flat_fn.flatten_in(*args)
        jac = jac_fn(*flat_args)
        if jac.ndim > 2:
            jac = jac.squeeze()
        return jnp.trace(jac, axis1=-2, axis2=-1)
    return _divergence


def scan_with_init_state(fn, init, n, length):
    """
    jnp.lax.scan doesnt return the initial state.
    This function does.
    """
    _,  xs = lax.scan(fn, init, n[:-1], length=length-1)
    return tree_map(lambda i, x: jnp.concatenate([i[None, ...], x], axis=0), init, xs)

def zip_arrays(a, b):
    """
    Given two arrays of the same shape (n, ...).
    Return a single array of shape (2 * n, ...).
    Where the values are zipped together yielding [a_0, b_0, a_1, b_1, ..., a_n, b_n]
    """
    assert a.shape == b.shape
    return jnp.concatenate([a[None, ...], b[None, ...]], axis=0).reshape(-1, *a.shape[1:], order='F')

def scan_wrapper(fn):
    def _fn(acc, x):
        y = fn(x, acc)
        return y, y
    return _fn