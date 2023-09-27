import numpy as np

class ConstantHazard:

    def __init__(self, p):
        self.l = 1/p

    def H(self, t):
        return self.l
    
    def log1mH(self, t):
        return np.log((1-self.l)*np.ones(t))

    def logH(self, t):
        return np.log(self.l*np.ones(t))
