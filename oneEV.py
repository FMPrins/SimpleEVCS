import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EVChargingEnv(gym.Env): 
    metadata = {"render_modes": ["human"]}

    def __init__(self, t_max=100, i_max=1, voltage=230.0, capacity=1000.0, delta_t=0.25):
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


