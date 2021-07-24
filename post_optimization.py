from model import Model
from simulation import Simulation

import matplotlib.pyplot as plt
from statistics import mean, median, stdev
from tqdm import tqdm

def simulate(model: Model, f: dict, y: 'list[float]', precision: int = -1):
    simulation = Simulation(f, y, model, precision = precision)
    tx_rewards = []

    for _ in tqdm(range(1000)):
        tx_rewards.append(simulation.run())
        
    print(f"(Analyzing: average reward per unit time over entire simulation)\n"+
          f"MEAN: {round(mean(tx_rewards), 4)}, " + 
          f"MEDIAN: {round(median(tx_rewards), 4)}, " +
          f"STDEV: {round(stdev(tx_rewards), 4)}"
    )
    
    plt.hist(tx_rewards, bins = 200)
    plt.show()

def confirm(msg: str) -> bool:
    """
    Ask the user for confirmation.
    """
    res = input(msg + " (Y/n) > ")
    if res == 'Y' or res == 'y' or res == 'yes' or res == 'Yes' or res == "":
        return True
    return False