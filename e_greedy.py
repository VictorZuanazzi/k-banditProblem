"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström 
"""
import random
from agent import *
import numpy as np

class E_greedy(Agent):

	def __init__(self, timesteps, epsilon = 0.1):

		super().__init__(timesteps)
		self.epsilon = epsilon

		print("e-greedy created: RED, epsilon =", epsilon)

	def chooseAction(self):
		# Decide whether to do a random move or pick the greedy move
		if(random.randint(0, 100) < (self.epsilon * 100)):
			# do random move
			action = random.randint(0, self.k - 1)
		else:
			# do greedy move
			action = np.argmax(self.estimates)
		return action