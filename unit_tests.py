from simulation import Simulation
from parameters import Parameters, validate_transmit_strategy, \
    validate_jammer_strategy, get_default_parameters
from model import Model

def test_create_parameters():

    params = get_default_parameters()

    print(repr(params))
    print(params)

def test_validate_jammer_strategy():
    
    params = get_default_parameters()
    y = create_demo_jammer_strategy(params)
    validate_jammer_strategy(params, y)

def create_demo_jammer_strategy(params: Parameters):
    y = [0, 1]
    for _ in range(params.m - 1):
        y.append(0)

    return y

def test_validate_transmit_strategy():

    params = get_default_parameters()
    model = Model(params)
    f = create_demo_transmit_strategy(params, model)
    validate_transmit_strategy(params, f, model.state_space, 
        model.action_space)

def create_demo_transmit_strategy(params: Parameters, model: Model):
    f = {}

    for state in model.state_space:
        f[state] = {}
        for action in model.action_space:
            if state != "j":
                if action == f"s{params.m}":
                    f[state][action] = 1
                else:
                    f[state][action] = 0
            else:
                if action == f"h{params.m}":
                    f[state][action] = 1
                else:
                    f[state][action] = 0

    return f

def test_run_simulation():
    
    params = get_default_parameters()
    model = Model(params)

    f = create_demo_transmit_strategy(params, model)
    y = create_demo_jammer_strategy(params)

    simulation = Simulation(f, y, params, debug=True)
    tx_reward = simulation.run()

    print(f"Transmitter reward: {tx_reward}")

def main():
    # test_create_parameters()
    # test_validate_jammer_strategy()
    # test_validate_transmit_strategy()
    test_run_simulation()

if __name__ == "__main__":
    main()