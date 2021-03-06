# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr
import random as r
from datetime import datetime

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):#sigma=0.2
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        #self.seed = int(datetime.now().strftime('%S%f'))
        #np.random.seed(self.seed)
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self, x):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        #dx = self.theta * (self.mu - x) + np.array([r.gauss(0,self.sigma),r.gauss(0,self.sigma),r.gauss(0,self.sigma),r.gauss(0,self.sigma)]) #thread safe
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.noise(1))
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
