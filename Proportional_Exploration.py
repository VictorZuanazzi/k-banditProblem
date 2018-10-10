# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:09:06 2018

@author: Victor Zuanazzi
"""

import random
from agent import *
import numpy as np
from k_bandit import *

class Proportional_exploration(Agent):
    
    def __init__(self, timesteps, k):
        super().__init__(timesteps)
        self.k = k
        self.probabilities = self.init_probabilities()
        self.estimates = 0
        print("Proportional Exploration created: Yellow")

    def init_probabilities(self):
        prob = []
        for i in range(self.k):
            prob.append(1/self.k)
        return prob
    
    def update_probabilities(self):
        lower = min(self.estimates) -1
        total = 0
        for estimate in self.estimates:
            total += estimate - lower
            
        for i in range(self.k):
            self.probabilities[i] = (self.estimates[i]-lower)/total
    
    def chooseAction(self):
        self.update_probabilities()
        while True:
            action = np.random.randint(0,self.k)
            if (np.random.uniform(0,1) < self.probabilities[action]):
                return action
    
if __name__ == '__main__':
    k =10
    timesteps = 10
    prop = Proportional_exploration(timesteps, 10) #create agent, initialize agent, initialize agent-type
    bandit = K_Bandit(k) #<-- how many arms are there?
    prop.initAgent(k)
    
    for t in range(1, timesteps + 1): # We do 1000 actions for each bandit
        print("t: ", t)
        print(prop.estimates)
        print(prop.probabilities)
        action = prop.chooseAction()
        reward = bandit.play(action)
        prop.updateTimestep(t - 1, reward)
        prop.updateAction(reward, action)
        
