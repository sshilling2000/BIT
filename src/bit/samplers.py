#see https://emcee.readthedocs.io/en/stable/tutorials/line/$0
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

#basic model - to be replaced with more complex models (or user-specified models)
def linear_model(theta, x):
    m, b = theta
    return m * x + b

#log prior prob.
def log_prior(theta):
    m, b = theta
    if -10 < m < 10 and -10 < b < 10:
        return 0.0
    return -np.inf

#log prob
def log_likelihood(theta, x, y, yerr):
    model = linear_model(theta, x)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2)

#posterior prob
def log_prob(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

#running the sampler
def run_sampler(log_prob_fn, initial_pos, n_steps=2000, n_walkers=32, args=()):
    """
    Run emcee sampler.

    Parameters
    ----------
    log_prob_fn : function
        Function(theta, *args) -> log probability
    initial_pos : array-like
        Initial guess for parameters
    """

    ndim = len(initial_pos)

    # small random perturbation around initial guess
    pos = initial_pos + 1e-4 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_prob_fn,
        args=args
    )

    sampler.run_mcmc(pos, n_steps, progress=True)

    return sampler

#wrap the everything and run the sampler - this is our function that we provide for users
def fit(log_prob_fn, initial_guess, args=(), burn_in=500):
    """
    High-level convenience function.

    Returns flattened posterior samples.
    """

    sampler = run_sampler(log_prob_fn, initial_guess, args=args)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    return samples