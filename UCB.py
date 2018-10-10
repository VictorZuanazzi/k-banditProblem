"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström 
"""
import random
from agent import *
import numpy as np

class UCB(Agent):

	def __init__(self, timesteps, c):
		
		super().__init__(timesteps)
		self.c = c

		print("UCB created: BLUE, c =", c)

	def correctActionCnt(self):
		for i in range(self.k):
			self.action_cnt[i] += 0.000001 #<-- this avoids division by 0 in the first action

	def chooseAction(self, t):
		# this is done in one statement because I want to show off,
		return np.argmax(np.asarray(self.estimates) + \
						self.c * np.sqrt(np.log(t) / (2 * np.asarray(self.action_cnt))) \
						)