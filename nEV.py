import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiEVChargingEnv(gym.Env): 
    metadata = {"render_modes": ["human"]}

    def __init__(self, t_max=100, n_evs = 2, i_max=10, p_max = 10, voltage=230.0, capacity=1000.0, delta_t=0.25):
        super().__init__()

        # --- Physical Parameters ---
        self.T = t_max          # Total timesteps in episode
        self.I_max = i_max      # Max Current (Amps)
        self.n_evs = n_evs      # Number of charging ports
        self.P_max = p_max      # Total available current (Amps)
        self.V = voltage        # Voltage (Volts)
        self.C = capacity       # Battery Capacity (kWh)
        self.delta_t = delta_t  # Hours per timestep

        # --- Action Space ---
        # The agent chooses a discrete current value from 0 to I_max
        self.action_space = spaces.MultiDiscrete((self.I_max + 1) ** self.n_evs)
        
        # --- Observation Space ---
        # Defined as [Occupancy, SOC] 
        # Using MultiDiscrete for easy indexing in your Q-table
        self.observation_space = spaces.MultiDiscrete([3] * self.n_evs)

        # --- Internal State Variables ---
        self._current_t = 0
        self._state = np.zeros(self.n_evs, dtype=np.int32) 
        self._soc = np.zeros(self.n_evs, dtype=np.float32)

    def _decode_action(self, index):
        base = self.I_max + 1
        action = np.zeros(self.n_evs, dtype=int)

        for i in range(self.n_evs - 1, -1, -1):
            action[i] = index % base
            index //= base

        return action

    def get_observation(self):
        return self._state.copy()

    def _arrival_occurred(self):
        """Probability of an EV arriving given the port is empty."""
        # This matches your p_a(t) logic
        p_a = 0.2 #if 8 <= (self._current_t % 24) <= 18 else 0.05
        return np.random.random() < p_a

    def _departure_occurred(self, soc):
        """Probability of an EV departing given the port is occupied."""
        # This matches your p_d(SOC) logic
        p_d = 0.01 + (0.2 * (soc / 100.0))
        return np.random.random() < p_d

    def reset(self, seed=None, options=None):
        """Resets environment to the start of an episode."""
        super().reset(seed=seed)
        self._current_t = 0
        self._state[:] = 0
        self._soc[:] = 0.0
        return self.get_observation(), {}

    def step(self, action_index):
        """
        Executes one timestep in the environment.
        1. Process physical charging if an EV is present.
        2. Check for stochastic departure.
        3. If empty, check for stochastic arrival.
        """
        self._current_t += 1
        reward = 0.0
        
        action = self._decode_action(action_index)

        if np.sum(action) > int(self.P_max):
            reward -= 0

        for ev in range(self.n_evs):

            if self._state[ev] == 1:
                energy = self.V * action[ev] * self.delta_t
                self._soc[ev] = min(100, self._soc[ev] + (energy / self.C) * 100)

                # If fully charged
                if self._soc[ev] >= 100:
                    self._state[ev] = 2

                # Stochastic departure
                elif self._departure_occurred(self._soc[ev]):
                    reward += self._soc[ev]
                    self._state[ev] = 0
                    self._soc[ev] = 0.0

            elif self._state[ev] == 2:
                if self._departure_occurred(100):
                    reward += self._soc[ev]
                    self._state[ev] = 0
                    self._soc[ev] = 0.0
                
            elif self._state[ev] == 0:
                if self._arrival_occurred():
                    self._state[ev] = 1
                    self._soc[ev] = np.random.random() * 50

        # Check if episode is over
        terminated = False
        truncated = self._current_t >= self.T

        return self.get_observation(), reward, terminated, truncated, {}


