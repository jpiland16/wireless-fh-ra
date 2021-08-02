from optimize import optimize_game, ROUND_PRECISION
from parameters import Parameters
from simulation import Simulation

from statistics import mean, median , stdev
import pickle

def figures_2_and_3():

    fh_ra_rates = Parameters().rates
    fh_only_6_rates = [6]
    fh_only_24_rates = [24]
    fh_only_54_rates = [54]

    variants = [fh_ra_rates]

    names = [
        "Joint FH and RA",
        "FH only, Rate = 54 Mbps",
        "FH only, Rate = 24 Mbps",
        "FH only, Rate = 6 Mbps"
    ]

    results = {}

    for i, rates in enumerate(variants):
        results[names[i]] = {}

        for k in range(3, 6):
            
            print(f"Optimizing:     {names[i]} for k = {k}")

            params = Parameters(rates = rates, k = k, m = len(rates) - 1)
            model, f, y = optimize_game(params)

            simulation = Simulation(f, y, model, 
                precision = ROUND_PRECISION - 2)
            tx_rewards = []
            tx_successes = []

            for _ in range(1000):
                reward, success = simulation.run()
                tx_rewards.append(reward)
                tx_successes.append(success)
                
            results[names[i]][str(k)] = {
                "rewards": {
                    "mean": mean(tx_rewards),
                    "median": median(tx_rewards),
                    "stdev": stdev(tx_rewards)
                }, 
                "successes": {
                    "mean": mean(tx_successes),
                    "median": median(tx_successes),
                    "stdev": stdev(tx_successes)
                }
            }

            print(f"Optimization of {names[i]} for k = {k} complete.")
            
    pickle.dump(results, open("results.pickle", "wb"))

def main():
    figures_2_and_3()

if __name__ == "__main__":
    main()