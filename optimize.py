from markov import QTable
from model import Model
from parameters import Parameters, get_default_parameters

from scipy.optimize import minimize, LinearConstraint
import numpy as np

DELTA = 0.9
CUTOFF = 0.001

def convert_strategies_to_list(f: dict, y: 'list[float]'):
    vector = []
    for state in f:
        for action in f[state]:
            vector.append(f[state][action])
    for prob in y:
        vector.append(prob)
    
    return vector

def convert_list_to_strategies(params: Parameters, vector: 'list'):
    f = {}
    y = []
    model = Model(params)
    i = 0

    for state in model.state_space:
        f[state] = {}
        for action in model.action_space:
            f[state][action] = vector[i]
            i += 1
    
    for _ in range(params.m + 1):
        y.append(vector[i])
        i += 1

    return f, y

def best_transmitter_value(params: Parameters, model: Model, state: str,
        f: dict, y: 'list[float]', fraction: float = DELTA):
    """
    Labeled as V_1 in Equation 22 of the paper.
    """
    if fraction < CUTOFF:
        return 0

    return max(
        np.dot(model.get_reward_matrix(state), y) 

         + fraction * np.dot(model.get_transition_matrix(state, 
             lambda new_state: best_transmitter_value(params, model, new_state,
                f, y, fraction * DELTA)
        ), y)
    )

def best_jammer_value(params: Parameters, model: Model, state: str,
        f: dict, y: 'list[float]', fraction: float = DELTA):
    """
    Labeled as V_2 in Equation 22 of the paper.
    """
    if fraction < CUTOFF:
        return 0

    action_probs = [-f[state][action] for action in f[state]]

    return max(
        np.dot(action_probs, model.get_reward_matrix(state)) 
         + fraction * np.dot(action_probs, model.get_transition_matrix(state, 
             lambda new_state: best_jammer_value(params, model, new_state,
                f, y, fraction * DELTA)
        ))
    )

def objective_function(x, *params_tuple):
    params = Parameters.get_from_tuple(params_tuple)
    f, y = convert_list_to_strategies(params, x)
    model = Model(params)

    v1 = lambda x: best_transmitter_value(params, model, x, f, y)
    v2 = lambda x: best_jammer_value(params, model, x, f, y)

    return sum([v1(state) + v2(state) for state in model.state_space])

def create_constraints(model: Model, vec_size: int):
    constraints = []
    action_count = len(model.action_space)
    vector_offset = 0

    for _ in model.state_space:
        # Ensure the transmitter's action probabilities sum to 1 for each state
        coeffs = [1 if i >= vector_offset and i - vector_offset < action_count 
            else 0 for i in range(vec_size)
        ]
        constraints.append(LinearConstraint(coeffs, 1, 1))
        vector_offset += action_count
    
    # Ensure the jammer's probabilities sum to 1
    coeffs = [1 if i >= vector_offset else 0 for i in range(vec_size)]
    constraints.append(LinearConstraint(coeffs, 1, 1))

def create_bounds(vec_size: int):
    bounds = []
    for _ in range(vec_size):
        bounds.append((0, 1))

def create_random_strategies(params: Parameters, model: Model):
    q_table = QTable(model.state_space, model.action_space)
    q_table.epsilon = 1 # Used to create completely random strategy
    rate_count = params.m + 1
    y = [1 / rate_count for _ in range(rate_count)]
    return q_table, y 

def find_equilibrium(params: Parameters):
    
    params_tuple = params.convert_to_tuple()
    model = Model(params)
    
    f, y = create_random_strategies(params, model)
    x0 = convert_strategies_to_list(f, y)

    constraints = create_constraints(model, len(x0))
    bounds = create_bounds(len(x0))

    return minimize(objective_function, x0, args=params_tuple, 
        bounds=bounds, constraints=constraints)

def main():
    params = get_default_parameters()
    eq = find_equilibrium(params)
    print(eq)    

if __name__ == "__main__":
    main()