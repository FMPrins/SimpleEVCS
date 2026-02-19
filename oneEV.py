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

class EVChargingEnv(gym.Env): 
    metadata = {"render_modes": ["human"]}

    def __init__(self, t_max=100, i_max=10, voltage=230.0, capacity=1000.0, delta_t=0.25):
        super().__init__()

        # --- Physical Parameters ---
        self.T = t_max          # Total timesteps in episode
        self.I_max = i_max      # Max Current (Amps)
        self.V = voltage        # Voltage (Volts)
        self.C = capacity       # Battery Capacity (kWh)
        self.delta_t = delta_t  # Hours per timestep

        # --- Action Space ---
        # The agent chooses a discrete current value from 0 to I_max
        self.action_space = spaces.Discrete(self.I_max + 1)
        
        # --- Observation Space ---
        # Defined as [Occupancy, SOC] 
        # Using MultiDiscrete for easy indexing in your Q-table
        self.observation_space = spaces.Discrete(3)

        # --- Internal State Variables ---
        self._current_t = 0
        self._z = 0             # Binary occupancy
        self._soc = 0.0         # Float for precision in internal calculation

    def get_observation(self):
        """Converts internal float SOC to integer for the agent's observation."""

        if self._z == 0:
            state = 0
        elif self._soc < 100:
            state = 1
        else:
            state = 2
        return np.array(state, dtype=np.int32)

    def _arrival_occurred(self):
        """Probability of an EV arriving given the port is empty."""
        # This matches your p_a(t) logic
        p_a = 0.2 #if 8 <= (self._current_t % 24) <= 18 else 0.05
        return np.random.random() < p_a

    def _departure_occurred(self):
        """Probability of an EV departing given the port is occupied."""
        # This matches your p_d(SOC) logic
        p_d = 0.01 + (0.2 * (self._soc / 100.0))
        return np.random.random() < p_d

    def reset(self, seed=None, options=None):
        """Resets environment to the start of an episode."""
        super().reset(seed=seed)
        self._current_t = 0
        self._z = 0
        self._soc = 0.0
        return self.get_observation(), {}

    def step(self, action):
        """
        Executes one timestep in the environment.
        1. Process physical charging if an EV is present.
        2. Check for stochastic departure.
        3. If empty, check for stochastic arrival.
        """
        self._current_t += 1
        reward = 0.0
        
        # Scenario A: Charging Port is Occupied
        if self._z == 1:
            # Update SOC based on the agent's current choice
            # Formula: SOC_new = SOC_old + (Energy_Received / Capacity) * 100
            energy_kw = (self.V * float(action) * self.delta_t)
            self._soc = min(100, self._soc + (energy_kw / self.C) * 100)

            # Check if vehicle departs AFTER charging this step
            if self._departure_occurred():
                reward = self._soc  # Reward is final SOC at departure
                self._z = 0
                self._soc = 0.0

        # Scenario B: Charging Port is Empty
        else:
            if self._arrival_occurred():
                self._z = 1
                self._soc = np.random.random() * 50  # New arrival starts at random % below 50 precent
                #print(self._soc)

        # Check if episode is over
        terminated = False
        truncated = self._current_t >= self.T

        return self.get_observation(), reward, terminated, truncated, {}

def objective(trial):
    # 2. Initialize a W&B run for this specific trial
    lr = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
    gamma = trial.suggest_float("discount_factor", 0.8, 0.99)
    eps = 0.01 #trial.suggest_float("epsilon", 0.001, 0.1)
    
    # 2. Create a descriptive name
    # Format: "run_LR-0.01_G-0.95_ED-0.995"
    run_name = f"trial_{trial.number}_LR-{lr:.3f}_G-{gamma:.2f}_E-{eps:.4f}"

    # 3. Initialize W&B with the custom name
    run = wandb.init(
        project="ev-charging-optimization",
        group="optuna-study",
        name=run_name,  # <--- This sets the name in the UI
        config={
            "learning_rate": lr,
            "discount_factor": gamma,
            "epsilon": eps,
            "trial_num": trial.number,
            "num_episodes": 5000,
        },
        reinit=True
    )
    config = wandb.config

    env = EVChargingEnv(t_max=100)
    agent = Qlearningagent(
        state_space_size=3, 
        action_space_size=env.action_space.n,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon,
        epsilon_decay=1
    )

    total_rewards = []
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        state = int(obs)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            next_state = int(next_obs)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        #agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        total_rewards.append(episode_reward)

        # 3. Log metrics to W&B
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            best_action_state_1 = np.argmax(agent.q_table[1])
            
            wandb.log({
                "episode": episode + 1,
                "avg_reward_100": avg_reward,
                "epsilon": agent.epsilon,
                "best_action_state_1": best_action_state_1
            })

            # Report back to Optuna for pruning
            # trial.report(avg_reward, episode)
            # if trial.should_prune():
            #     run.finish(exit_code=1) # Mark as failed/pruned in W&B
            #     raise optuna.exceptions.TrialPruned()

    final_performance = np.mean(total_rewards[-100:])
    run.finish() # 4. Close the run
    return final_performance

if __name__ == "__main__":
    # Ensure you are logged in: wandb login
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

