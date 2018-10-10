"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström 
"""
from k_bandit import *

class Agent:

	def __init__(self, timesteps):
		self.timesteps = [0] * timesteps 

	def updateAction(self, newR, action):
		self.estimates[action] = self.estimates[action] + \
			(1.0/(self.action_cnt[action] + 1)) * \
			(newR - self.estimates[action])
		self.action_cnt[action] += 1

	def updateTimestep(self, step, newR):
		self.timesteps[step] += newR

	def initAgent(self, k):
		self.k = k
		self.estimates = [0] * k
		self.action_cnt = [0] * k

if __name__ == '__main__':

	bandit = K_Bandit(3)
	agent = Agent(10)
	action = 2 #idx
	agent.initAgent(3)

	for i in range(15):

		agent.updateAction(bandit.play(action), action)
		print("Estimate for a =", action, " --> ", agent.estimates[action])
