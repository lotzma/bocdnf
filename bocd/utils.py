import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
from scipy.special import gammaln


def generate_data_const_var(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps

def generate_data(alpha0, beta0, kappa0, mu0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob`
    """
    data  = []
    cps   = []
    meanx = mu0
    taux = stats.gamma.rvs(alpha0, scale=1/beta0)
    for t in range(0, T):
        if np.random.random() < cp_prob:
            taux = stats.gamma.rvs(alpha0, scale=1/beta0)
            meanx = np.random.normal(mu0, 1/(kappa0*taux))
            cps.append(t)
        data.append(np.random.normal(meanx, 1/taux))
    return data, cps

def plot_posterior(T, data, cps, bocd):
    R = bocd.get_P()
    est_cp = bocd.get_cp()

    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted', linewidth=5)
        ax2.axvline(cp, c='red', ls='dotted', linewidth=5)
    # Estimated changepoints
    for cp in est_cp:
        ax1.axvline(cp, c='blue', ls='dashed', linewidth=3)
        ax2.axvline(cp, c='blue', ls='dashed', linewidth=3)

    plt.tight_layout()
    plt.show()

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def gammanormal(x,y,m,k,a,b):
    return(stats.norm.pdf(x,loc=m,scale=1/np.sqrt(k*y))*stats.gamma.pdf(y,a,0,1/b))

def loginvchi2(x, nu, tausquared):
    prod = nu*tausquared
    return (nu/2)*(np.log(prod)-np.log(2))-gammaln(nu/2)-prod/(2*x)+(1+nu/2)*np.log(x)