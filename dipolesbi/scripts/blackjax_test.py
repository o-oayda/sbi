import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import blackjax
from anesthetic import NestedSamples

rng_key = jax.random.PRNGKey(42)
num_live = 100
num_delete = 50

num_data = 15
x = jnp.linspace(-2.0, 2.0, num_data)
true = {'m': 2.0, 'c': 1.0, 'sigma': 0.5}

key, rng_key = jax.random.split(rng_key)
noise = true['sigma'] * jax.random.normal(key, (num_data,))
y = true['m'] * x + true['c'] + noise

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, y, yerr=true['sigma'], fmt="o", label="Observed data", color='black')
ax.plot(x, true['m'] * x + true['c'], '--', label="True model", color='red', alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Linear Model: Bayesian Parameter Estimation")
plt.show()

def line_loglikelihood(params):
    """Log-likelihood for linear model with Gaussian noise."""
    m, c, sigma = params["m"], params["c"], params["sigma"]
    y_model = m * x + c
    # Vectorized normal log-likelihood
    return jax.scipy.stats.multivariate_normal.logpdf(y, y_model, sigma**2)

prior_bounds = {
    "m": (-5.0, 5.0),      # slope
    "c": (-5.0, 5.0),      # intercept  
    "sigma": (0.1, 2.0),   # noise level (positive)
}

num_dims = len(prior_bounds)
num_inner_steps = num_dims * 5

rng_key, prior_key = jax.random.split(rng_key)
particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)

nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=line_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)
print(f"Initialized nested sampler with {num_live} live points")

init_fn = jax.jit(nested_sampler.init)
step_fn = jax.jit(nested_sampler.step)
print("Functions compiled - ready to run!")

print("Running nested sampling for line fitting...")
ns_start = time.time()
live = init_fn(particles)
dead = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not live.logZ_live - live.logZ < -3:  # Convergence criterion
        rng_key, subkey = jax.random.split(rng_key, 2)
        live, dead_info = step_fn(subkey, live)
        dead.append(dead_info)
        pbar.update(num_delete)

dead = blackjax.ns.utils.finalise(live, dead)
ns_time = time.time() - ns_start

columns = ["m", "c", "sigma"]
labels = [r"$m$", r"$c$", r"$\sigma$"]
data = jnp.vstack([dead.particles[key] for key in columns]).T

line_samples = NestedSamples(
    data,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=columns,
    labels=labels,
    logzero=jnp.nan,
)

print(f"Nested sampling runtime: {ns_time:.2f} seconds")
print(f"Log Evidence: {line_samples.logZ():.2f} ± {line_samples.logZ(100).std():.2f}")
print(f"True parameters: m={true['m']}, c={true['c']}, σ={true['sigma']}")
print(f"Posterior means: m={line_samples.m.mean():.2f}, c={line_samples.c.mean():.2f}, σ={line_samples.sigma.mean():.2f}")

# Create posterior corner plot with true values marked
kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
axes = line_samples.plot_2d(columns, kinds=kinds, label='Posterior')
axes.axlines(true, color='red', linestyle='--', alpha=0.8)
plt.suptitle("Line Fitting: Posterior Distributions")
plt.show()
