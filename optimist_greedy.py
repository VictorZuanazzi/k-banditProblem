"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström 
"""

import random
from agent import *
import numpy as np

class Optimist_greedy(Agent):

	def __init__(self, timesteps):

		super().__init__(timesteps)
		print("Optimist created: GREEN")

	def initOptimist(self, k, opt = 10):
		self.opt = opt 
		for i in range(k):
			self.estimates[i] = opt

	def chooseAction(self):
		return np.argmax(self.estimates)

if __name__ == '__main__':
	opt = Optimist_greedy(100) #create agent, initialize agent, initialize agent-type
	opt.initAgent(3)
	opt.initOptimist(3, 9)
	print(opt.estimates)