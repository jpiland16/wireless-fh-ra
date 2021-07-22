from parameters import Parameters, validate_jammer_strategy, \
    validate_transmit_strategy
from model import Model

p_max = 2
alpha = 1
sigma_squared = 0.01
p_recv = 1

class Simulation:
    def __init__(self, f: 'dict', y: 'list[float]', parameters: 
            Parameters = Parameters(
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
            rates = [6, 9, 12, 18, 24, 36, 48, 54]), initial_state: str = "j"):

        self.model = Model(parameters)

        validate_transmit_strategy(parameters, f, self.model.state_space, 
            self.model.action_space)
        validate_jammer_strategy(parameters, y)

        self.params = parameters
        self.f = f
        self.y = y
        self.state = initial_state