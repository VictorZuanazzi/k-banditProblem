"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström 
"""
import random
import numpy as np

class K_Bandit:

	def __init__(self, k):
			
		self.std = 0.1
		self.k = k
		# This ensures that each agent will play against the same k-armed
		# bandit and that each bandit plays the same sequence of moves 
		random.seed(k)
		np.random.seed(k)

		# Initialize a list of random means between -5 and 5
		self.means = [random.randint(-5, 5) for i in range(k)]
		# print("True means of bandit:", self.means)

	# This function returns the stochastic utility of playing 'arm'
	def play(self, arm):
		
		return np.random.normal(self.means[arm], self.std)
		


if __name__ == '__main__':
	obj = K_Bandit(5)
	print(obj.means)
	for i in range(11):
		print(obj.play(0))