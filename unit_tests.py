from markov import QTable
from simulation import Simulation
from parameters import Parameters, get_default_parameters
from optimize import convert_strategies_to_list, convert_list_to_strategies
from model import Model, validate_transmit_strategy, validate_jammer_strategy

from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import stdev, median, mean

def test_create_parameters():

    params = get_default_parameters()

    print(repr(params))
    print(params)

def test_validate_jammer_strategy():
    
    params = get_default_parameters()
    model = Model(params)
    y = create_demo_jammer_strategy(model)
    validate_jammer_strategy(model, y)

def create_demo_jammer_strategy(model: Model):
    params = model.params

    y = [0, 1]
    for _ in range(params.m - 1):
        y.append(0)

    return y

def test_validate_transmit_strategy():

    params = get_default_parameters()
    model = Model(params)
    f = create_demo_transmit_strategy(model)
    validate_transmit_strategy(model, f)

def create_demo_transmit_strategy(model: Model):
    params = model.params

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

    f = create_demo_transmit_strategy(model)
    y = create_demo_jammer_strategy(model)

    simulation = Simulation(f, y, model)
    tx_reward = simulation.run()

    print(f"Transmitter reward: {tx_reward}")

def test_multiple_simulation():
    
    params = get_default_parameters()
    model = Model(params)

    f = create_demo_transmit_strategy(model)
    y = create_demo_jammer_strategy(model)

    simulation = Simulation(f, y, model)
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

    qtable = QTable(model)
    validate_transmit_strategy(model, qtable)

def test_convert_parameters():

    params = get_default_parameters()
    same_params = Parameters.get_from_tuple(params.convert_to_tuple())

    print(params)
    print(same_params)

def test_convert_strategies():

    params =  get_default_parameters()
    model = Model(params)

    f = create_demo_transmit_strategy(model)
    y = create_demo_jammer_strategy(model)

    fp, yp = convert_list_to_strategies(model, 
        convert_strategies_to_list(f, y))

    def compare(a, b, name):
        if a == b:
            print(f"Conversion of {name} was successful.")
        else:
            print(f"WARNING: Conversion of {name} was not successful!")
            print(a)
            print(b)

    compare(f, fp, "f")
    compare(y, yp, "y")

def test_random_strategies():

    params = get_default_parameters()
    model = Model(params)

    qtable = QTable(model)
    qtable.epsilon = 1
    y = [1 / (params.m + 1) for _ in range(params.m + 1)]

    sim = Simulation(qtable, y, model)
    print(sim.run())

def main():
    # test_create_parameters()
    # test_validate_jammer_strategy()
    # test_validate_transmit_strategy()
    # test_run_simulation()
    # test_multiple_simulation()
    # test_qtable_as_f()
    # test_convert_parameters()
    # test_convert_strategies()
    test_random_strategies()

if __name__ == "__main__":
    main()