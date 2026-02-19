import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import pandas as pd
import wandb
import optuna

class Qlearningagent:

    def __init__(self, state_space_size = 4, action_space_size = 2, learning_rate=0, discount_factor=0.99, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.9995
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: (num_states × num_actions)
        self.q_table = np.zeros((state_space_size, action_space_size))

    def select_action(self, state):
        # State is now an integer instead of [1,1]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space_size)

        else:
            action = np.argmax(self.q_table[state])
        return action


    def update(self, state, action, reward, next_state, done):
    # This implements the Bellman Optimality Equation for Q-Learning:
    # Q(S, A) <- Q(S, A) + alpha * [ R + gamma * max_a' Q(S', a') - Q(S, A) ]
        
        # Calculate the Target Q-value: R + gamma * max_a' Q(S', a')
        if done:
            target = reward
        else:
            max_q_next = np.max(self.q_table[next_state, :])
            target = reward + self.discount_factor * max_q_next
 
        current_q_value = self.q_table[state, action]
        
        # Calculate the TD Error 
        td_error = target - current_q_value
        
        # Apply the update rule: 
        self.q_table[state, action] += self.learning_rate * td_error


        # if reward > 0: # Focus on when a vehicle departs with value
        #     print(f"   [UPDATE] State {state} Action {action} -> Reward: {reward:.2f}, TD Error: {td_error:.4f}")

        return self.q_table
