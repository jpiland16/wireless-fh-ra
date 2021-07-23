from markov import QTable
from simulation import Simulation
from parameters import Parameters, validate_transmit_strategy, \
    validate_jammer_strategy, get_default_parameters
from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import stdev, median, mean

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

    simulation = Simulation(f, y, params)
    tx_reward = simulation.run()

    print(f"Transmitter reward: {tx_reward}")

def test_multiple_simulation():
    
    params = get_default_parameters()
    model = Model(params)

    f = create_demo_transmit_strategy(params, model)
    y = create_demo_jammer_strategy(params)

    simulation = Simulation(f, y, params)
    tx_rewards = []

    for _ in tqdm(range(2000)):
        tx_rewards.append(simulation.run())
        
    print(f"(Analyzing: average reward per unit time over entire simulation)\n"+
          f"MEAN: {round(mean(tx_rewards), 4)}, " + 
          f"MEDIAN: {round(median(tx_rewards), 4)}, " +
          f"STDEV: {round(stdev(tx_rewards), 4)}"
    )
    
    plt.hist(tx_rewards, bins = 200)
    plt.show()

def test_qtable_as_f():

    params = get_default_parameters()
    model = Model(params)

    qtable = QTable(model.state_space, model.action_space)
    validate_transmit_strategy(params, qtable, model.state_space, 
        model.action_space)

def main():
    # test_create_parameters()
    # test_validate_jammer_strategy()
    # test_validate_transmit_strategy()
    # test_run_simulation()
    # test_multiple_simulation()
    test_qtable_as_f()

if __name__ == "__main__":
    main()