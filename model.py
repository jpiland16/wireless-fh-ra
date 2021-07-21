import math
import numpy as np
from parameters import Parameters

p_max = 2
alpha = 1
sigma_squared = 0.01
p_recv = 1

class Model:
    def __init__(self, parameters: Parameters = Parameters(
            k = 4,
            m = 7,
            p_avg = 0.83 * p_max,
            p_max = p_max,
            c = 50,
            l = 25,
            n = 1, 
            alpha = alpha,
            sigma_squared = sigma_squared,
            p_recv = p_recv,
            rates = [6, 9, 12, 18, 24, 36, 48, 54]) ):

        self.params = self.params
        self.state_space = self.get_state_space()
        self.action_space = self.get_action_space()

    def get_state_space(self):
        
        states = ["j"]
        for i in range(math.floor(self.params.k / self.params.m) ):
            states.append(str(i))
            
        return states
        
    def get_action_space(self):
        return [ "s" + str(rate) for rate in self.params.rates ] + \
            [ "h" + str(rate) for rate in self.params.rates]
        

    def get_immediate_transmitter_payoff(self, action: str, 
            jammer_power_index: int, next_state: str):
        """
        Listed as U(., a1, a2, x') in the paper.
        """

        r = int(action[1:])

        # Equation 8
        if (next_state == "j" and action[0] == "h" 
                and jammer_power_index > self.params.m - r):
            return - self.params.l - self.params.c

        elif (next_state == "1" and action[0] == "h" 
                and jammer_power_index <= self.params.m - r):
            return r - self.params.c

        elif (next_state == "j" and action[0] == "s"
                and jammer_power_index > self.params.m - r):
            return - self.params.l

        elif (next_state != "j" and action[0] == "s"
                and jammer_power_index <= self.params.m - r):
            return r
        
        else:
            return 0

    def get_transition_probabilities(self, state: str, action: str,
            jammer_power_index: int):
        """
        Listed as P(x'|x, a1, a2) in the paper. This function provides a 
        dict containing values for all x' rather than a value for a single x'.
        """
                
        probs = {k: 0 for k in self.state_space}
        
        # Equation 9
        if state == "j" and action[0] == "h":
            probs["j"] = self.params.n / (self.params.k - 1) \
                if jammer_power_index > self.params.m - int(action[1:]) else 0
            probs["1"] = 1 - probs["j"]
        
        # Equation 11 (same as #9) ???
        elif action[0] == "h":
            probs["j"] = self.params.n / (self.params.k - 1) \
            if jammer_power_index > self.params.m - int(action[1:]) else 0
            probs["1"] = 1 - probs["j"]
        
        # Equation 12   
        else:
            x = int(state)
            r = int(action[1:])
            sinr_single_attack = self.params.p_recv / ( self.params.alpha * 
                self.params.n * self.params.p_jam[jammer_power_index]
                + self.params.sigma_squared)
            
            p_discover_next = self.params.n / (self.params.k - 
                self.params.n * x)
            p_single_channel_attack = self.params.m * x / self.params.k
            
            probs["j"] = p_discover_next + p_single_channel_attack \
                if x < self.params.k / self.params.m and \
                jammer_power_index > self.params.m - r else (
                    p_single_channel_attack if x < self.params.k / self.params.m 
                    and sinr_single_attack < self.params.sinr_limits[r]
                    else 0
                )
            probs[str(x + 1)] = 1 - probs["j"]

    def get_immediate_transmitter_reward(self, state: str, action: str, 
            jammer_power_index: int):     
        """
        Listed as r(x, a1, a2) in the paper.
        """ 
        
        transition_probabilities = self.get_transition_probabilities(
            state, action, jammer_power_index)

        # Equation 13
        return sum([self.get_immediate_transmitter_payoff(action, 
            jammer_power_index, x_prime) * transition_probabilities[x_prime] 
            for x_prime in self.state_space])

    def get_reward_matrix(self, state: str):
        """
        Listed as R(x) in the paper.
        """
        return np.array([
                [self.get_immediate_transmitter_reward(state, action, 
                jammer_power_index) for jammer_power_index 
                in range(0, self.params.m + 1)] 
            for action in self.action_space])

    def get_transition_matrix(self, state: str, value_function: callable):
        """
        Listed as T(x) in the paper.
        """

        matrix = []

        for action in self.action_space:
            matrix.append([])
            for jammer_power_index in range(0, self.params.m + 1):
                transition_probabilities = self.get_transition_probabilities(
                    state, action, jammer_power_index)
                matrix[-1].append(sum([transition_probabilities[x_prime] 
                    * value_function(x_prime) 
                    for x_prime in self.state_space]))

        return np.array(matrix)