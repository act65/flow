import fire
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from functools import partial

import flax.linen as nn
from flax.training import train_state
from jax.flatten_util import ravel_pytree
import optax

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from bayes.posterior import FlowBasedPosterior, PRNGKeyManager
from sinterp.interpolants import OneSidedLinear
from bayes.map import find_map_from_samples
# jax.config.update('jax_disable_jit', True)
jax.config.update("jax_debug_nans", True)

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron for digit classification."""
    num_hidden: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        # We return logits, softmax will be handled in the loss function
        return x
    
"""
A posterior over weights feels awkward as there are many symmetries in the weights. f(\theta, x) == f(T(\theta), x)

(there is a lot of redundancy?!)
"""

# --- Main Simulation Logic ---
def run_simulation(distillation_threshold: int, num_observations: int):
    """
    Runs the Bayesian inference simulation on the digits dataset.
    """
    key_manager = PRNGKeyManager(seed=0)

    # 1. Load and prepare data
    digits = load_digits()
    X = (digits.data / 16.0).astype(jnp.float32) # Normalize pixel values
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data loaded. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 2. Initialize the model to get parameter shapes and utilities
    model = SimpleMLP(num_hidden=32, num_outputs=10)
    dummy_input = jnp.ones((1, X_train.shape[1]))
    params = model.init(key_manager.split(), dummy_input)['params']
    
    # Utility to flatten/unflatten the model's parameters (theta)
    flat_params, unflatten_fn = ravel_pytree(params)
    DIM = len(flat_params)
    print(f"MLP initialized. Parameter dimension (theta): {DIM}")

    # 3. Define the log-likelihood function for the posterior
    def build_total_log_likelihood_and_grad(observations):
        """
        Builds a function that computes the gradient of the total log-likelihood.
        An observation is a tuple containing a batch of (images, labels).
        """
        # We expect observations to be a list of tuples, where each tuple contains
        x_data = jnp.stack([obs[0] for obs in observations])
        y_data = jnp.stack([obs[1] for obs in observations])

        def total_log_likelihood(theta_flat, data):
            x_batch = data[0]
            y_batch = data[1]

            """Calculates log p(y | x, theta) summed over a batch."""
            # Unflatten the parameter vector into a PyTree for the Flax model
            theta_tree = unflatten_fn(theta_flat)
            
            # Get model predictions (logits)
            logits = vmap(model.apply, in_axes=(None, 0))({'params': theta_tree}, x_batch)
            
            # Calculate log-softmax probabilities
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            
            # Get the log-probability of the true labels
            # This uses advanced integer indexing
            true_log_probs = log_probs[jnp.arange(len(y_batch)), y_batch]
            
            return jnp.sum(true_log_probs)

        # Return the JIT-compiled gradient function
        grad_fn = jax.grad(total_log_likelihood, argnums=0)
        return grad_fn, (x_data, y_data)


    # 4. Set up the Flow-Based Posterior
    interpolator = OneSidedLinear()
    posterior = FlowBasedPosterior(
        dim=DIM,
        key_manager=key_manager,
        interpolator=interpolator,
        build_total_log_likelihood_and_grad=build_total_log_likelihood_and_grad,
        distillation_threshold=distillation_threshold,
        num_train_steps=200,
        n_steps=25
    )

    # 5. Define the evaluation function
    @jit
    def eval_accuracy(flat_params, x_test, y_test):
        """Calculates test accuracy for a given set of flattened parameters."""
        params_tree = unflatten_fn(flat_params)
        logits = vmap(model.apply, in_axes=(None, 0))({'params': params_tree}, x_test)
        predictions = jnp.argmax(logits, axis=-1)
        assert predictions.shape == y_test.shape
        return jnp.mean(predictions == y_test)

    # 6. Run the simulation and evaluation loop
    accuracies = []
    n_obs = []
    
    for i in range(num_observations):
        x = X_train[i]
        y = y_train[i]

        # Add the new observation to the posterior
        print(f"\n--- Adding Observation #{i+1} ---")
        posterior.add_observation((x, y))

        # Evaluate every k observations
        if i % 20 == 0:
            # Find the MAP estimate from the current posterior
            theta_map_flat, _ = find_map_from_samples(
                posterior, key_manager, num_samples=20, num_steps=20
            )
            
            # Calculate and store test accuracy
            print("now eval acc")
            accuracy = eval_accuracy(theta_map_flat, X_test, y_test)
            accuracies.append(accuracy)
            
            print(f"\n>>> After {i} observations, Test Accuracy = {accuracy:.4f} <<<")

            n_obs.append(i)

    return accuracies, n_obs

def plot(accuracies, n_obs, label):
    """Plots the test accuracy vs. number of observations."""
    plt.plot(n_obs, accuracies, marker='o', linestyle='-', label=label)

def main(num_observations: int = 200, dt1: int = 300, dt2: int = 50):
    """
    Main function to run simulations and plot results.
    
    Args:
        num_observations: The total number of training data points to process.
        dt1: Distillation threshold for the first run.
        dt2: Distillation threshold for the second run.
    """
    plt.figure(figsize=(10, 6))

    print("\n" + "="*50)
    print(f"STARTING SIMULATION 1 (Distillation Threshold = {dt1})")
    print("="*50)
    accuracies1, n_obs1 = run_simulation(
        distillation_threshold=dt1, num_observations=num_observations
    )
    plot(accuracies1, n_obs1, label=f'Distill Threshold = {dt1}')

    # print("\n" + "="*50)
    # print(f"STARTING SIMULATION 2 (Distillation Threshold = {dt2})")
    # print("="*50)
    # accuracies2, n_obs2 = run_simulation(
    #     distillation_threshold=dt2, num_observations=num_observations
    # )
    # plot(accuracies2, n_obs2, label=f'Distill Threshold = {dt2}')
    
    # Final plot formatting
    plt.xlabel("Number of Observations")
    plt.ylabel("Test Accuracy on MAP Estimate")
    plt.title("Test Accuracy vs. Number of Observations on Digits Dataset")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Save the plot
    filename = 'digits_test_accuracy.png'
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")
    # plt.show()


if __name__ == '__main__':
    fire.Fire(main)