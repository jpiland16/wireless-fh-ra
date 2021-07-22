from parameters import validate_transmit_strategy, validate_jammer_strategy, \
    get_default_parameters
from model import Model

def test_create_parameters():

    params = get_default_parameters()

    print(repr(params))
    print(params)

def test_validate_jammer_strategy():
    
    params = get_default_parameters()
    y = [1]
    for _ in range(params.m):
        y.append(0)
    
    validate_jammer_strategy(params, y)

def test_validate_transmit_strategy():

    params = get_default_parameters()
    model = Model(params)
    f = {}

    for state in model.state_space:
        f[state] = {}
        for action in model.action_space:
            if action == f"s{params.m}":
                f[state][action] = 1
            else:
                f[state][action] = 0

    validate_transmit_strategy(params, f, model.state_space, 
        model.action_space)

def main():
    test_create_parameters()
    test_validate_jammer_strategy()
    test_validate_transmit_strategy()

if __name__ == "__main__":
    main()