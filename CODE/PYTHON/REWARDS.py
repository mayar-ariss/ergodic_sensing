class REWARDS:
    def __init__(self, alpha, beta, energy, accuracy, max_energy, max_delay):
        """
        Initialize the REWARDS class with parameters.
        
        Args:
            alpha (float): Weight for the delay penalty.
            beta (float): Weight for the state recognition reward.
            energy (float): Energy consumption of the sensor group.
            accuracy (float): Accuracy of the sensor group in recognizing states.
            max_energy (float): Maximum energy cost among sensor groups.
            max_delay (float): Maximum possible delay.
        """
        self.alpha = alpha
        self.beta = beta
        self.energy = energy
        self.accuracy = accuracy
        self.max_energy = max_energy
        self.max_delay = max_delay

    def compute_energy_reward(self, energy_cost):
        """
        Compute the energy component of the reward function.
        
        Args:
            energy_cost (float): Energy cost of the sensing decision.

        Returns:
            float: Normalized energy penalty.
        """
        return -energy_cost / self.max_energy

    def compute_delay_reward(self, delta_t):
        """
        Compute the delay component of the reward function.
        
        Args:
            delta_t (float): Time elapsed since the last sensing decision.

        Returns:
            float: Combined delay penalty and state recognition reward.
        """
        if delta_t == 0:
            return 0, 0  # Avoid division by zero
        delay_penalty = -self.alpha * delta_t
        recognition_reward = self.beta * (self.accuracy / delta_t)
        return delay_penalty, recognition_reward

    def compute_survival_probability(self, time, threshold, epsilon=1e-3):
        """
        Calculate survival probability based on time and threshold.
        
        Args:
            time (float): Current time step.
            threshold (float): Transition threshold time.
            epsilon (float): Minimum survival probability near the threshold.

        Returns:
            float: Survival probability.
        """
        if time >= threshold:
            return epsilon
        return 1 - (time / threshold)

    def get_instantaneous_reward(self, delta_t, energy_cost, survival_prob, state_change):
        """
        Calculate the instantaneous reward for a sensing decision.
        
        Args:
            delta_t (float): Time elapsed since the last sensing decision.
            energy_cost (float): Energy cost of the sensing decision.
            survival_prob (float): Probability of survival of the current state.
            state_change (bool): Whether a state change is detected.

        Returns:
            float: Instantaneous reward.
        """
        energy_reward = self.compute_energy_reward(energy_cost)
        delay_penalty, recognition_reward = self.compute_delay_reward(delta_t)

        # State change vs. no state change
        if state_change:
            reward = energy_reward + delay_penalty + recognition_reward
        else:
            reward = energy_reward

        # Incorporate survival probability
        return survival_prob * reward + (1 - survival_prob) * energy_reward

    def get_Mult_Reward(self, time, current_action, next_action, threshold):
        """
        Calculate the reward for transitioning between actions at a given time step.
        
        Args:
            time (float): Current time step in the schedule.
            current_action (str): Current action ("Sense" or "NotSense").
            next_action (str): Next action to take ("Sense" or "NotSense").
            threshold (float): Transition threshold time.

        Returns:
            float: Reward for the transition.
        """
        delta_t = 1 if current_action == "NotSense" else 0  # Simplified delay
        energy_cost = self.energy if next_action == "Sense" else 0
        survival_prob = self.compute_survival_probability(time, threshold)
        state_change = next_action == "Sense"  # Assuming sensing detects state change

        return self.get_instantaneous_reward(delta_t, energy_cost, survival_prob, state_change)
