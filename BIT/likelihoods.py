import numpy as np #pip install numpy
import emcee # https://emcee.readthedocs.io/en/stable/user/install/ conda install -c conda-forge emcee

#class for basic gaussian likelihoods
class GaussianLikelihood:

    #constructor to initialize the data and model function
    def __init__(self, x, y, yerr, model, model_params, model_params_priors):
        """
        Store the data needed for likelihood evaluations.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.yerr = np.asarray(yerr)
        self.model=model #must be a function that accepts x (independent variable) as first argument 
        self.model_params=model_params
        self.model_params_priors=model_params_priors

    #function to check that the data types are correct
    def check_types(self):
        """
        Check that the data types are correct.
        """
        assert isinstance(self.x, np.ndarray), "x must be a numpy array"
        assert isinstance(self.y, np.ndarray), "y must be a numpy array"
        assert isinstance(self.yerr, np.ndarray), "yerr must be a numpy array"
        assert callable(self.model), "model must be a callable function"
        assert isinstance(self.model_kwargs, dict), "model_params must be a dictionary"
        assert isinstance(self.model_params, dict), "model_params must be a dictionary"
        assert isinstance(self.model_params_priors, dict), "model_params_priors must be a dictionary"

        
    #function to calculate the model output given the parameters
    def model_output(self):
        """
        Evaluate the model at the given parameters.
        """
        return self.model(self.x, **self.model_params)

    #function to calculate the residuals between the model output and the data
    def residuals(self):
        """
        Compute the residuals between the data and the model.
        """
        return self.model_output() - self.y

    #function to evaluate the log prior
    def log_prior(self):
        """
        Uniform priors on parameters.
        Check we are within the prior bounds, if not then return -inf, if all params pass then return 0.0
        """
        for param in self.model_params.keys():
            low_limit, high_limit = self.model_params_priors[param]
            if not (low_limit < self.model_params[param] < high_limit):
                return -np.inf

        return 0.0

    
    #function to evaluate the log likelihood 
    #https://redback.readthedocs.io/en/latest/_modules/redback/likelihoods.html#GaussianLikelihood.__init__
    def log_likelihood(self): 
        """
        Gaussian log-likelihood.
        """
        return np.sum(- (self.residuals() / self.yerr) ** 2 / 2 - np.log(2 * np.pi * self.yerr ** 2) / 2)

   
    #function to evaluate the log posterior
    def log_probability(self):
        """
        Posterior = prior + likelihood.
        Evaluate the log prior and log likelihood, and return the sum.
        """
        prior_prob = self.log_prior()
        if not np.isfinite(prior_prob):
            return -np.inf

        return prior_prob + self.log_likelihood()