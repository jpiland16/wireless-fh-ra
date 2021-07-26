import math
import numpy as np
from parameters import Parameters, validate_param

class Model:
    def __init__(self, params: Parameters = Parameters()):

        self.params = params
        self.state_space = self.get_state_space()
        self.action_space = self.get_action_space()

        # Do calculations now to avoid repetition
        self.transition_probabilities = {
            state: {
                action: [
                    self.get_transition_probabilities(state, action, pj) \
                    for pj in range(self.params.m + 1)
                ] for action in self.action_space
            } for state in self.state_space
        }

        self.transmitter_payoffs = {
            action: [
                {
                    new_state: self.get_immediate_transmitter_payoff(action, 
                        pj, new_state)
                    for new_state in self.state_space
                } for pj in range(self.params.m + 1)
            ] for action in self.action_space
        }

        self.transmitter_rewards = {
            state: {
                action: [
                    self.get_immediate_transmitter_reward(state, action, pj) \
                    for pj in range(self.params.m + 1)
                ] for action in self.action_space
            } for state in self.state_space
        }

        self.reward_matrices = {
            state: self.get_reward_matrix(state) for state in self.state_space
        }

    def get_state_space(self):
        
        states = ["j"]
        for i in range(math.ceil(self.params.k / self.params.n) ):
            states.append(str(i + 1))
            
        return states
        
    def get_action_space(self):
        return [ "s" + str(i) for i, _ in enumerate(self.params.rates) ] + \
            [ "h" + str(i) for i, _ in enumerate(self.params.rates)]
        

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
            try:
                # Messy. Need to check this later. TODO
                x = int(state) 
            except ValueError:
                x = 0

            r = int(action[1:])
            sinr_single_attack = self.params.p_recv / ( self.params.alpha * 
                self.params.n * self.params.p_jam[jammer_power_index]
                + self.params.sigma_squared)
            
            # TODO the max thing below could also be wrong. Trying to avoid 
            # errors to see what happens.
            p_discover_next = self.params.n / max(1, (self.params.k - 
                self.params.n * x))
            p_single_channel_attack = self.params.n * x / self.params.k
            
            probs["j"] = p_discover_next + p_single_channel_attack \
                if x < self.params.k / self.params.n and \
                jammer_power_index > self.params.m - r else (
                    p_single_channel_attack if x < self.params.k / self.params.n 
                    and sinr_single_attack < self.params.sinr_limits[r]
                    else 0
                )
            probs[str(x + 1)] = 1 - probs["j"]

        return probs

    def get_immediate_transmitter_reward(self, state: str, action: str, 
            jammer_power_index: int):     
        """
        Listed as r(x, a1, a2) in the paper.
        """ 
        
        transition_probabilities = (
            self.transition_probabilities[state][action][jammer_power_index])

        # Equation 13
        return sum([self.transmitter_payoffs[action][jammer_power_index] \
                [x_prime] * transition_probabilities[x_prime] 
            for x_prime in self.state_space])

    def get_reward_matrix(self, state: str):
        """
        Listed as R(x) in the paper.
        """
        return np.array([
                [self.transmitter_rewards[state][action][jammer_power_index] \
                    for jammer_power_index in range(0, self.params.m + 1)] 
            for action in self.action_space])

    def get_transition_matrix(self, state: str, value_function: callable):
        """
        Listed as T(x) in the paper.
        """

        matrix = []
        next_state_values = {x_prime: value_function(x_prime) for x_prime 
            in self.state_space}

        for action in self.action_space:
            matrix.append([])
            for jammer_power_index in range(0, self.params.m + 1):
                transition_probabilities = (self.transition_probabilities \
                    [state][action][jammer_power_index])
                matrix[-1].append(sum([transition_probabilities[x_prime] 
                    * next_state_values[x_prime] 
                    for x_prime in self.state_space]))

        return np.array(matrix)

################################## VALIDATION ##################################

def validate_transmit_strategy(model: Model, f: dict,
        precision: int = -1):
    """
    Ensures that the strategy of the transmitter is valid, and throws a 
    ValueError otherwise. If precision is a non-negative integer, then
    sums are first rounded to that precision before beuing evaluated.
    """
    params = model.params
    state_space = model.state_space
    action_space = model.action_space

    validate_param("state space", "size", math.ceil(params.k 
        / params.n) + 1, len(state_space))
    
    validate_param("action space", "size", (params.m + 1) * 2, 
        len(action_space))

    def transmit_validate(p_name: str, expected, actual):
        validate_param("transmit strategy", p_name, expected, actual)
    
    for state in state_space:
        action_p_sum = 0
        for action in action_space:
            p = f[state][action]
            transmit_validate(f"0 <= f[{state}][{action}] <= 1", 
                True, 0 <= p and p <= 1)
            action_p_sum += p
        transmit_validate(f"sum of f[\"{state}\"]", 1, action_p_sum if 
            precision < 0 else round(action_p_sum, precision))
        transmit_validate(f"number of actions in f[\"{state}\"]", 
            len(action_space), len(f[state]))

    transmit_validate("number of states in f", len(state_space), len(f))


def validate_jammer_strategy(model: Model, y: 'list[float]', 
        precision: int = -1):
    """
    Ensures that the strategy of the jammer is valid, and throws a 
    ValueError otherwise. If precision is a non-negative integer, then
    sums are first rounded to that precision before beuing evaluated.
    """
    params = model.params

    def jammer_validate(p_name: str, expected, actual):
        validate_param("jammer strategy", p_name, expected, actual)    

    jammer_validate("number of elements", params.m + 1, len(y))

    for i, yi in enumerate(y):
        jammer_validate(f"0 <= y[{i}] <= 1", True, 0 <= yi and yi <= 1)

    jammer_validate("sum of elements", 1, sum(y) if precision < 0 else 
        round(sum(y), precision))

    avg_power = np.dot(params.p_jam, y)

    jammer_validate("power constraint satisfied", True, (avg_power
        if precision < 0 else round(avg_power, precision)) <= params.p_avg)