import numpy as np
import polars as pl
from scipy.special import logsumexp
from sklearn.metrics import mutual_info_score
from .utils import moving_average


class BOCD:
    """Bayesian Online Changepoint Detection
    """
    def __init__(self, model, hazard, max_depth=1000, threshold=10, window=2):
        self.T = max_depth
        self.thresh = threshold
        self.window = window
        self.logP = -np.inf*np.ones((self.T+1, self.T+1))
        self.logP[0,0] = 0
        self.P = np.zeros((self.T+1, self.T+1))
        self.P[0,0] = 1
        self.model = model
        self.hazard = hazard
        self.t = 1
        self.log_message = np.array([0])
        self.changepoints = []

    def step(self, x):
        log_pred = self.model.get_log_pred_prob(self.t, x)
        temp = log_pred + self.log_message[:self.t]
        self.log_message = np.append(logsumexp(self.hazard.logH(self.t) + temp),
                                     self.hazard.log1mH(self.t) + temp)
        temp = self.log_message - logsumexp(self.log_message)
        self.logP[self.t,:self.t+1] = temp
        self.P[self.t,:self.t+1] = np.exp(temp)

        # See if changepoint detected (could be done better)
        cp_occurred = False
        if self.t > self.window:
            ma1 = np.diff(self.P[self.t-1,:])
            ma2 = np.diff(self.P[self.t,:])
            if np.argmax(ma1)-np.argmax(ma2) >= self.thresh:
                cp_occurred = True
                self.changepoints.append(self.t-1)

        # Update parameters and exit
        self.model.update(x)
        self.t += 1
        return self.t-1, cp_occurred

    def get_P(self):
        return np.exp(self.logP)
    
    def get_cp(self):
        return self.changepoints