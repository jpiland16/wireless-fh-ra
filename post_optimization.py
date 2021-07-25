from model import Model
from simulation import Simulation

import matplotlib.pyplot as plt
from statistics import mean, median, stdev
from tqdm import tqdm

def simulate(model: Model, f: dict, y: 'list[float]', precision: int = -1):
    
    simulation = Simulation(f, y, model, precision = precision)
    tx_rewards = []
    tx_successes = []

    for _ in tqdm(range(2000)):
        reward, success = simulation.run()
        tx_rewards.append(reward)
        tx_successes.append(success)
        
    print(f"Average reward per unit time over entire simulation\n"+
          f"MEAN: {round(mean(tx_rewards), 4)}, " + 
          f"MEDIAN: {round(median(tx_rewards), 4)}, " +
          f"STDEV: {round(stdev(tx_rewards), 4)}"
    )

    print(f"Success rate\n"+
          f"MEAN: {round(mean(tx_successes), 4)}, " + 
          f"MEDIAN: {round(median(tx_successes), 4)}, " +
          f"STDEV: {round(stdev(tx_successes), 4)}"
    )
    
    fig, (ax1, ax2) = plt.subplots(ncols = 2)

    ax1.set_title("Reward")
    ax1.hist(tx_rewards, bins = 200)
    ax2.set_title("Success rate")
    ax2.hist(tx_successes, bins = 200)

    plt.show()

def confirm(msg: str) -> bool:
    """
    Ask the user for confirmation.
    """
    res = input(msg + " (Y/n) > ")
    if res == 'Y' or res == 'y' or res == 'yes' or res == 'Yes' or res == "":
        return True
    return False