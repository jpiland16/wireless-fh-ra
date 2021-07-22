from parameters import Parameters, validate_jammer_strategy, \
    validate_transmit_strategy, get_default_parameters
from model import Model


class Simulation:
    def __init__(self, f: 'dict', y: 'list[float]', parameters: 
            Parameters = get_default_parameters(), initial_state: str = "j"):

        self.model = Model(parameters)

        validate_transmit_strategy(parameters, f, self.model.state_space, 
            self.model.action_space)
        validate_jammer_strategy(parameters, y)

        self.params = parameters
        self.f = f
        self.y = y
        self.state = initial_state