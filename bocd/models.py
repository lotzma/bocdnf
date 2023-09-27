import numpy as np
import scipy.stats as stats
from scipy.stats.distributions import chi2
from .utils import gammanormal, loginvchi2


class GaussianMean:
    """Gaussian with fixed variance and changing mean
    """
    
    def __init__(self, mean0, var0, varx, maxt=None):
        self.mean0 = mean0
        self.var0 = var0
        self.prec0 = 1/self.var0
        self.varx = varx
        self.precx = 1/self.varx
        self.maxt = maxt
        self.mean_params = np.array([self.mean0])
        self.prec_params = np.array([self.prec0])
        self.t = len(self.mean_params)

    def get_log_pred_prob(self, t, x):
        """Compute posterior predictive probabilities
        """
        assert(t <= self.t)
        post_means = self.mean_params[:t]
        vars = 1./self.prec_params + self.varx
        post_stds = np.sqrt(vars[:t])
        return stats.norm(post_means, post_stds).logpdf(x)
    
    def update(self, x):
        """Upon observing a new data point x at time t, update all run length 
        hypotheses.
        """
        # Update precisions
        new_precs = self.prec_params + self.precx
        self.prec_params = np.append([self.prec0], new_precs)
        # Update means
        new_means = (self.mean_params * self.prec_params[:-1] + \
                     (x * self.precx)) / new_precs
        self.mean_params = np.append([self.mean0], new_means)
        self.t += 1

class GaussianVar:
    """Gaussian with fixed mean and changing variance.
    """

    def __init__(self, mean0, var0, meanx):
        self.mean0 = mean0
        self.var0 = var0
        self.meanx = meanx



class GaussianNormalGamma:
    """Gaussian with changing mean and variance
    """

    def __init__(self, alpha0, beta0, kappa0, mu0, maxt=None):
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.mu0 = mu0
        self.maxt = maxt
        self.kappa_params = np.array([self.kappa0])
        self.alpha_params = np.array([self.alpha0])
        self.beta_params = np.array([self.beta0])
        self.mu_params = np.array([self.mu0])
        self.t = len(self.mu_params)

    def get_log_pred_prob(self, t, x):
        """Compute posterior predictive probabilities
        """
        assert(t <= self.t)
        mu_params = self.mu_params[:t]
        alpha_params = self.alpha_params[:t]
        sigma_params = (self.beta_params[:t]/self.alpha_params[:t])*(1+1/self.kappa_params[:t])
        return stats.t(2*alpha_params).logpdf((x-mu_params)/sigma_params)/sigma_params
    
    def update(self, x):
        """Upon observing a new data point x at time t, update all run length hypotheses.
        """
        # Update precisions
        new_kappas = self.kappa_params + 1
        self.kappa_params = np.append([self.kappa0], new_kappas)
        new_alphas = self.alpha_params + 0.5
        self.alpha_params = np.append([self.alpha0], new_alphas)
        new_mus = (self.mu_params*self.kappa_params+x)/(self.kappa_params+1)
        self.mu_params = np.append([self.mu0], new_mus)
        new_betas = self.beta_params + self.kappa_params*(x-self.mu_params[:-1])/(2*self.kappa_params+1)
        self.beta_params = np.append([self.beta0], new_betas)
        self.t += 1
        

class GaussianInverseChiSquared:
    """Gaussian with changing mean and variance
    """

    def __init__(self, mu0, kappa0, nu0, sigma0, maxt=None):
        self.kappa0 = kappa0
        self.mu0 = mu0
        self.nu0 = nu0
        self.sigma0 = sigma0
        self.maxt = maxt
        self.kappa_params = np.array([self.kappa0])
        self.mu_params = np.array([self.mu0])
        self.nu_params = np.array([self.nu0])
        self.sigma_params = np.array([self.sigma0])
        self.t = len(self.mu_params)

    def log_posterior(self, mu, sigma, mu0, nu0, sigma0, kappa0):
        nfactor = stats.norm(mu0, sigma/kappa0).logpdf(mu)
        cfactor = loginvchi2(sigma, nu0, sigma0)
        return nfactor+cfactor

    def get_log_pred_prob(self, t, x):
        """Compute posterior predictive probabilities
        """
        assert(t <= self.t)
        mu_params = self.mu_params[:t]
        alpha_params = self.alpha_params[:t]
        sigma_params = (self.beta_params[:t]/self.alpha_params[:t])*(1+1/self.kappa_params[:t])
        return stats.t(2*alpha_params).logpdf((x-mu_params)/sigma_params)/sigma_params
    
    def update(self, x):
        """Upon observing a new data point x at time t, update all run length hypotheses.
        """
        # Update precisions
        new_kappas = self.kappa_params + 1
        self.kappa_params = np.append([self.kappa0], new_kappas)
        new_alphas = self.alpha_params + 0.5
        self.alpha_params = np.append([self.alpha0], new_alphas)
        new_mus = (self.mu_params*self.kappa_params+x)/(self.kappa_params+1)
        self.mu_params = np.append([self.mu0], new_mus)
        new_betas = self.beta_params + self.kappa_params*(x-self.mu_params[:-1])/(2*self.kappa_params+1)
        self.beta_params = np.append([self.beta0], new_betas)
        self.t += 1