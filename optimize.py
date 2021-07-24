import time
from markov import QTable
from model import Model, validate_jammer_strategy, validate_transmit_strategy
from parameters import get_default_parameters
from post_optimization import confirm, simulate

from scipy.optimize import minimize, LinearConstraint
import numpy as np
from copy import deepcopy
from threading import Thread

DELTA = 0.9
TIME_AHEAD = 10 # How many timesteps ahead to consider (before ending recursion)
ROUND_PRECISION = 4 # Must be greater than or equal to 2 (see rounding in main)
GENTLE_STOPPING = True

stop_optimization = False
optimization_not_complete = True

class OptimizationProgress():
    def __init__(self):
        self.iterations = 0

    def __call__(self, x0):
        self.iterations += 1
        print(f"Iteration #{self.iterations} " + 
            f"(xk = {str(x0)[:15]}...)  \r", end="")

class StoppableFunction():
    def __init__(self, fun):
        self.function = fun
    def __call__(self, x):
        if stop_optimization:
            raise StopIteration
        self.last_input = x
        return self.function(x)

class MemoryFunctions():
    """
    Provides major performance improvements for the recursive functions
    as used in this game by remembering the output of V1(x) and V2(x) at 
    each depth. 

    Example:
        V1(j) --> new calculation
          V1(j) --> new calculation
          V1(1) --> new calculation
          .
          .
          .
        V1(1) --> new calculation
          V1(j) --> remembered
          V1(1) --> remembered
        
    I'm not completely sure, but I think this changes the objective function's
    time complexity from exponential to linear.
    """

    def __init__(self, model: Model):
        self.model = model
        self.function_count = 2

    def reset(self, f: dict, y: 'list[float]'):
        self.f = f
        self.y = y
        self.history = []
        for _ in range(2):
            self.history.append({x: {} for x in self.model.state_space})

    def get(self, funcId: int, x: str, depth: int = 0):
        try:
            return self.history[funcId][x][str(depth)]
        except KeyError:
            pass
        
        # We haven't done this calculation yet
        res = (best_transmitter_value(self, x, self.y, depth) 
            if funcId == 0 
            else best_jammer_value(self, x, self.f, depth))
        
        self.history[funcId][x][str(depth)] = res
        return res

def convert_strategies_to_list(f: dict, y: 'list[float]'):
    vector = []
    for state in f:
        for action in f[state]:
            vector.append(f[state][action])
    for prob in y:
        vector.append(prob)
    
    return vector

def convert_list_to_strategies(model: Model, vector: 'list'):
    f = {}
    y = []
    params = model.params
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

def best_transmitter_value(memfunc: MemoryFunctions, state: str,
        y: 'list[float]', exponent: int = 0):
    """
    Labeled as V_1 in Equation 22 of the paper.
    """
    model = memfunc.model

    if exponent > TIME_AHEAD:
        return 0

    return max(
        np.dot(model.reward_matrices[state], y) 

         + (DELTA ** exponent) * np.dot(model.get_transition_matrix(state, 
             lambda new_state: memfunc.get(0, new_state, exponent + 1)
        ), y)
    )

def best_jammer_value(memfunc: MemoryFunctions, state: str,
        f: dict, exponent: int = 0):
    """
    Labeled as V_2 in Equation 22 of the paper.
    """
    model = memfunc.model

    if exponent > TIME_AHEAD:
        return 0

    action_probs = [-f[state][action] for action in f[state]]

    return max(
        np.dot(action_probs, model.reward_matrices[state]) 
         + (DELTA ** exponent) * np.dot(action_probs, 
            model.get_transition_matrix(state, 
                lambda new_state: memfunc.get(1, new_state, exponent + 1)
            )
        )
    )

def objective_function(x, memfunc: MemoryFunctions):
    model = memfunc.model
    f, y = convert_list_to_strategies(model, x)

    memfunc.reset(f, y)

    return sum([memfunc.get(0, state) + memfunc.get(1, state) 
        for state in model.state_space])

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

    # Ensure the jammer meets its average power constraint
    coeffs = [0 for _ in range(vector_offset)]
    for p in model.params.p_jam:
        coeffs.append(p)
    constraints.append(LinearConstraint(coeffs, 0, model.params.p_avg))

    return constraints

def create_bounds(vec_size: int):
    bounds = []
    for _ in range(vec_size):
        bounds.append((0, 1))

    return bounds

def create_random_strategies(model: Model):
    params = model.params
    q_table = QTable(model.state_space, model.action_space)
    q_table.epsilon = 1 # Used to create completely random strategy
    rate_count = params.m + 1
    y = [1 / rate_count for _ in range(rate_count)]
    return q_table, y 

def find_equilibrium(model: Model):
    global optimization_not_complete
    
    f, y = create_random_strategies(model)
    x0 = convert_strategies_to_list(f, y)

    constraints = create_constraints(model, len(x0))
    bounds = create_bounds(len(x0))

    progress = OptimizationProgress()

    memfunc = MemoryFunctions(model)
    fun = StoppableFunction(lambda x: objective_function(x, memfunc))

    try:
        result = minimize(fun, x0, bounds=bounds, constraints=constraints, 
            callback=progress).x
    except StopIteration:
        result = fun.last_input
    
    optimization_not_complete = False
    return result

def round_strategies(f: dict, y: 'list[float]', decimal_places: int):
    """
    Rounds the strategies to the specified precision and returns the new 
    strategies. Uses deepcopy to avoid affecting the objects referenced
    by the input parameters.
    """
    f = deepcopy(f)
    y = deepcopy(y)

    for state in f:
        for action in f[state]:
            f[state][action] = round(f[state][action], decimal_places)
    for i in range(len(y)):
        y[i] = round(y[i], decimal_places)

    return f, y

def run_optimization():

    start_time = time.time()

    print("\nOptimizing the game... (CTRL-C to stop)")

    params = get_default_parameters()
    model = Model(params)
    eq = find_equilibrium(model)

    f, y = convert_list_to_strategies(model, eq)
    f, y = round_strategies(f, y, decimal_places = ROUND_PRECISION)

    print("\n\nTRANSMITTER STRATEGY: ")
    print(f)
    print("\nJAMMER STRATEGY: ")
    print(y)
    print()

    # validate_transmit_strategy(model, f, precision = ROUND_PRECISION - 2)
    # validate_jammer_strategy(model, y, precision = ROUND_PRECISION - 2)

    print(f"Elapsed time: {round(time.time() - start_time, 2)} seconds\n")

    if confirm("Run a simulation with these strategies?"):
        simulate(model, f, y, precision = ROUND_PRECISION - 2)

if __name__ == "__main__":

    if GENTLE_STOPPING:

        run = Thread(target=run_optimization)
        run.start()

        try:
            while optimization_not_complete:
                time.sleep(1)
                pass
        except KeyboardInterrupt:
            stop_optimization = True
            print("\nEnded process using KeyboardInterrupt")

        run.join()

    else:
        run_optimization()
