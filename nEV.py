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
        self.p_max = p_max      # Total available current (Amps)
        self.V = voltage        # Voltage (Volts)
        self.C = capacity       # Battery Capacity (kWh)
        self.delta_t = delta_t  # Hours per timestep

        # --- Action Space ---
        # The agent chooses a discrete current value from 0 to I_max
        self.action_space = spaces.MultiDiscrete([self.I_max + 1, self.I_max + 1])
        
        # --- Observation Space ---
        # Defined as [Occupancy, SOC] 
        # Using MultiDiscrete for easy indexing in your Q-table
        self.observation_space = spaces.MultiDiscrete([3,3])

        # --- Internal State Variables ---
        self._current_t = 0
        self._state = np.zeros(2, dtype=np.int32)  # [ev1_state, ev2_state]
        self._soc = np.zeros(2, dtype=np.float32)  # SOC for each EV

    # def get_observation(self):
    #     """Converts internal float SOC to integer for the agent's observation."""

    #     if self._z == 0:
    #         state = 0
    #     elif self._soc < 100:
    #         state = 1
    #     else:
    #         state = 2
    #     return np.array(state, dtype=np.int32)

    def get_observation(self):
        return self._state.copy()

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
        self._state[:] = 0
        self._soc[:] = 0.0
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
        

        i1, i2 = action

        if i1 + i2 > self.P_max:
            reward -= 10

        currents = [i1, i2]

        for ev in range(2):

            if self._state[ev] == 1:
                energy = self.V * currents[ev] * self.delta_t
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


