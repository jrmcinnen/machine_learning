"""
Reinforcement learning
"""

"""
Author:
Jere Mäkinen
Email:
jeremakinen98@gmail.com
"""

# Load OpenAI Gym and other necessary packages
import gym
import numpy as np

# Environment
env = gym.make("Taxi-v3")

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
alpha = 0.1
gamma = 0.6

# Number of training runs
epochs  = 100000

# Training
for i in range(epochs):
    state = env.reset()
    done = False
    
    while not done:
        # Select random action
        action = env.action_space.sample() 

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value


        state = next_state

print(f"Training finished after {epochs} training runs.\n")

# For computing the average total reward and number of actions
num_of_actions = 0
total_reward = 0
# Number of test runs
n = 10

# Testing
for j in range(n):
    state = env.reset()
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        total_reward += reward
        num_of_actions += 1

print(f"Results after {n} test run:")
print(f"Average number of actions per run: {num_of_actions / n}")
print(f'Average reward per run: {total_reward / n}')
        
        