from sinterp.couplings import IndependentCoupling, EMDCoupling, ConditionalCoupling

import jax.numpy as jnp 
from jax import random, grad, jit, vmap

import ot as pot

import matplotlib.pyplot as plt

# Assuming the EMDCoupling class above is in sinterp.couplings
from sinterp.couplings import EMDCoupling

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

def test_emd():
    key = random.PRNGKey(42)
    n = 500

    key, subkey = random.split(key)
    # Sort x to make the coupling visually obvious
    x = jnp.sort(random.normal(subkey, (n, 1)), axis=0)
    
    key, subkey = random.split(key)
    # Create a second distribution with a clear transformation (e.g., shift and scale)
    y = random.normal(subkey, (n, 1)) * 0.5 + 2.0

    # Use a small regularization value. A larger value would make the coupling more "blurry".
    coupling = EMDCoupling(reg=1e-3)

    key, subkey = random.split(key)
    x_hat, y_hat = coupling(subkey, x, y)

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    # Plot the original, uncoupled samples
    plt.subplot(1, 2, 1)
    # To show the original distributions, we can sort y as well for a side-by-side comparison
    plt.scatter(x, jnp.sort(y, axis=0), alpha=0.6)
    plt.title("Original Distributions (Sorted)")
    plt.xlabel("x (sorted)")
    plt.ylabel("y (sorted)")
    plt.grid(True)

    # Plot the samples after OT coupling
    plt.subplot(1, 2, 2)
    plt.scatter(x_hat, y_hat, alpha=0.6, c='r')
    plt.title("EMD Coupled Samples")
    plt.xlabel("x_hat (from x)")
    plt.ylabel("y_hat (from y)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()