#see https://emcee.readthedocs.io/en/stable/tutorials/line/$0
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

#basic model - to be replaced with more complex models (or user-specified models)
def linear_model(theta, x):
    """Linear model function for demo.

    Args:
        theta (list): List containing the slope (m) and intercept (b) parameters
        x (numpy.ndarray): Array of x-values (independent variable)

    Returns:
        numpy.ndarray: Array of y-values (dependent variable)
    """

    m, b = theta

    return m * x + b


#log prior prob.
def log_prior(theta):
    """Log prior probability function (hard coded for the linear model example).

    Args:
        theta (list): List containing the slope (m) and intercept (b) parameters

    Returns:
        float: Log prior probability
    """

    m, b = theta
    if -10 < m < 10 and -10 < b < 10:
        return 0.0
    
    return -np.inf


#log prob
def log_likelihood(theta, x, y, yerr):
    """Log likelihood function (hard coded for the linear model example).

    Args:
        theta (list): List containing the slope (m) and intercept (b) parameters
        x (numpy.ndarray): Array of x-values (independent variable)
        y (numpy.ndarray): Array of y-values (dependent variable)
        yerr (numpy.ndarray): Array of uncertainties on y-values

    Returns:
        float: Log likelihood
    """

    model = linear_model(theta, x)
    sigma2 = yerr**2

    return -0.5 * np.sum((y - model)**2 / sigma2)


#posterior prob
def log_prob(theta, x, y, yerr):
    """Posterior probability function (hard coded for the linear model example).

    Args:
        theta (list): List containing the slope (m) and intercept (b) parameters
        x (numpy.ndarray): Array of x-values (independent variable)
        y (numpy.ndarray): Array of y-values (dependent variable)
        yerr (numpy.ndarray): Array of uncertainties on y-values

    Returns:
        float: Posterior probability
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, x, y, yerr)


#running the sampler
def run_sampler(log_prob_fn, initial_guess, n_steps=2000, n_walkers=32, args=()):
    """Wrapper for running the sampler (hard coded for the linear model example).

    Args:
        log_prob_fn (function): Posterior probability function (users need to use the hard coded for the linear model example).
        initial_guess (list): Initial guesses for parameters (users need to use the hard coded for the linear model example).
        n_steps (int, optional): Number of steps to run the sampler. Defaults to 2000.
        n_walkers (int, optional): Number of walkers. Defaults to 32.
        args (tuple, required but coded as optional for now): This a tuple for (x, y, yerr).

    Returns:
        generated samples: Sampler object containing the generated samples.
    """

    ndim = len(initial_guess)

    # small random perturbation around initial guess
    pos = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)

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
    """High level convenience function that calls the sampler and returns flattened chains for each parameter

    Args:
        log_prob_fn (function): Posterior probability function (users need to use the hard coded for the linear model example).
        initial_guess (list): Initial guesses for parameters (users need to use the hard coded for the linear model example).
        args (tuple, required but coded as optional for now): This a tuple for (x, y, yerr).
        burn_in (int, optional): Number of steps to discard as burn-in. Defaults to 500.

    Returns:
        1D flattened samples for each parameter: list or array (not sure)
    """

    sampler = run_sampler(log_prob_fn, initial_guess, args=args)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    return samples