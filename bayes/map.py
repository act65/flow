from bayes.posterior import FlowBasedPosterior, PRNGKeyManager

class ParameterNet(nn.Module):
    """An MLP to parameterize a search variable x."""
    dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(features=self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.dim)(x)
        return x

def find_map_with_overparameterization(
    posterior: FlowBasedPosterior,
    key_manager: PRNGKeyManager,
    num_steps: int = 2000,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256
):
    """
    Finds the MAP estimate of a posterior by overparameterizing the search
    variable x with a neural network and optimizing its parameters.
    """
    print("\n--- Finding MAP Estimate via Overparameterization ---")
    dim = posterior.dim

    # 1. Define the reparameterizing network and its parameters
    param_net = ParameterNet(dim=dim, hidden_dim=hidden_dim)
    # A fixed, dummy input for the network
    dummy_input = jnp.zeros((1,))
    param_net_params = param_net.init(key_manager.split(), dummy_input)['params']

    # 2. Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(param_net_params)

    # 3. Define the loss function to be minimized
    # We want to MAXIMIZE posterior.log_prob(x), which is equivalent to
    # MINIMIZING -posterior.log_prob(x), where x = NN(theta).
    def loss_fn(params):
        # Generate the candidate x from the network
        x_candidate = param_net.apply({'params': params}, dummy_input)
        x_candidate = jnp.squeeze(x_candidate)
        return -posterior.log_prob(x_candidate)

    # 4. JIT-compile the training step for efficiency
    @jit
    def step(params, opt_state):
        loss_value, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # 5. Run the optimization loop
    for i in range(num_steps):
        param_net_params, opt_state, loss = step(param_net_params, opt_state)
        if i % 200 == 0:
            print(f"Step {i}, Negative Log Posterior: {loss:.4f}")

    # 6. Get the final MAP estimate by applying the optimized parameters
    x_map = param_net.apply({'params': param_net_params}, dummy_input)
    x_map = jnp.squeeze(x_map)
    final_log_prob = posterior.log_prob(x_map)

    print("--- MAP Finding Complete ---")
    print(f"Final Log Posterior: {final_log_prob:.4f}")
    print(f"Found MAP estimate x: {x_map}")

    return x_map, final_log_prob


if __name__ == '__main__':
    # 1. Setup
    DIM = 2
    SEED = 42
    key_manager = PRNGKeyManager(SEED)

    # 2. Define a "true" parameter for our likelihood to target
    true_x = jnp.array([2.5, -3.0])
    print(f"True parameter x to be found: {true_x}")

    # 3. Create a dummy likelihood function builder
    # This function's gradient will "guide" the posterior towards true_x
    def make_dummy_likelihood(target_x):
        def build_total_log_likelihood_and_grad(observations):
            def total_log_likelihood_grad_fn(x):
                # Gradient of log N(y|x,I) w.r.t. x is (y-x)
                # We assume a single observation y=target_x
                return target_x - x
            return None, total_log_likelihood_grad_fn
        return build_total_log_likelihood_and_grad

    # 4. Instantiate and "train" the posterior
    # The distillation process will absorb the information from the likelihood
    # (i.e., that the posterior should be centered around true_x)
    posterior = FlowBasedPosterior(
        build_total_log_likelihood_and_grad=make_dummy_likelihood(true_x),
        dim=DIM,
        key_manager=key_manager,
        interpolator=OneSidedLinear(),
        distillation_threshold=1 # Distill immediately for this demo
    )
    # Add a dummy observation to trigger the distillation process
    posterior.add_observation({'data': 1})

    # 5. Find the MAP estimate using the overparameterization method
    x_map_found, log_prob_at_map = find_map_with_overparameterization(
        posterior,
        key_manager
    )