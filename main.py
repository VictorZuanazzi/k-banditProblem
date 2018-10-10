"""
Created on Sun Oct  7 21:09:06 2018

@author: Arvid Lindström, Victor Zuanazzi
"""

from k_bandit import *
from optimist_greedy import *
from e_greedy import *
from UCB import * 
from Proportional_Exploration import *
import random
import matplotlib.pyplot as plt

# Global variables
timesteps = 1000

# Greedy-Epsilon agent
greedy = E_greedy(timesteps, epsilon = 0.05) #<-- chosen arbitrarily
for k in range(5, 21):
	bandit = K_Bandit(k) #<-- how many arms are there?
	greedy.initAgent(k) #<-- how many actions can I choose from?
	for t in range(1, timesteps + 1): # We do 1000 actions for each bandit
		action = greedy.chooseAction()
		reward = bandit.play(action)
		greedy.updateTimestep(t - 1, reward)
		greedy.updateAction(reward, action)

# Greedy-Optimist agent
optimist = Optimist_greedy(timesteps)
for k in range(5, 21):
	bandit = K_Bandit(k)
	optimist.initAgent(k)
	optimist.initOptimist(k, opt = 10) #<-- this is done every time because this code is not very good
	for t in range(1, timesteps + 1): 
		action = optimist.chooseAction()
		reward = bandit.play(action)
		optimist.updateTimestep(t - 1, reward)
		optimist.updateAction(reward, action)

# Upper Confidence Bound agent
ucb = UCB(timesteps, c = 1)
for k in range(5, 21):
	bandit = K_Bandit(k) 
	ucb.initAgent(k) 
	ucb.correctActionCnt()
	for t in range(1, timesteps + 1): # We do 1000 actions for each bandit
		action = ucb.chooseAction(t) #<-- dependent on time-step 't'
		reward = bandit.play(action)
		ucb.updateTimestep(t - 1, reward)
		ucb.updateAction(reward, action)

# Proportional Exploration (apparently that is a very bad idea!)
prop_e = Proportional_exploration(timesteps, k) 
#Proportional Exploration assigns probabilities to actions that are proportional
#to its expected pay offs.
for k in range(5, 21):
	bandit = K_Bandit(k) #<-- how many arms are there?
	prop_e.initAgent(k) #<-- how many actions can I choose from?
	for t in range(1, timesteps + 1): # We do 1000 actions for each bandit
		action = prop_e.chooseAction()
		reward = bandit.play(action)
		prop_e.updateTimestep(t - 1, reward)
		prop_e.updateAction(reward, action)


# Create average timesteps list for each agent
greedy_avg = (np.asarray(greedy.timesteps) / 16.0).tolist()
optimist_avg = (np.asarray(optimist.timesteps) / 16.0).tolist()
ucb_avg = (np.asarray(ucb.timesteps) / 16.0).tolist()
prop_avg =(np.asarray(prop_e.timesteps) / 16.0).tolist()

# Plot each agent's average timesteps
plt.plot(range(1, timesteps + 1), greedy_avg, 'r')
plt.title("Four types of agents")
plt.xlabel("Timesteps")
plt.ylabel("Average reward")
# plt.xlim(-10, 1000)
# plt.ylim(0, 6)

plt.plot(range(1, timesteps + 1), optimist_avg, 'g')
plt.plot(range(1, timesteps + 1), ucb_avg, 'b')
plt.plot(range(1, timesteps + 1), prop_avg, 'y')
plt.show()