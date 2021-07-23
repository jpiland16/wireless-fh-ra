from model import Model
from parameters import Parameters, get_default_parameters

from scipy.optimize import minimize, LinearConstraint, Bounds
import numpy as np

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


def find_equilibrium(params: Parameters):
    pass


def main():
    params = get_default_parameters()
    eq = find_equilibrium(params)
    print(eq)    

if __name__ == "__main__":
    main()