import numpy as np
import math

def validate_param(msg: str, p_name: str, expected, actual):
        """
        Raises an error with message `msg` when the parameter of name `p_name` 
        has a value (`actual`) that differs from the expected value.
        """
        if expected != actual:
            raise ValueError(f"Invalid {msg}. Expected " + 
                             f"{p_name} = {str(expected)}, got "
                             f"{str(actual)}.")

class Parameters:
    def __init__(self, k: int, m: int, p_avg: float, p_max: float, c: int, 
            l: int, n: int, alpha: float, sigma_squared: float, p_recv: float,
            rates: 'list[int]'):
        """
        Parameters:
         - `k`: the number of channels
         - `m`: the number of available rates, minus 1 (there are m + 1 rates)
         - `p_avg`: the average power constraint on the jammer (units?)
         - `p_max`: the maximum power of the jammer on any given channel 
                (units?)
         - `c`: the hopping cost for the transmitter (Mbps)
         - `l`: the jamming cost for the transmitter (Mbps)
         - `n`: the number of channels that the jammer can block/listen to at 
                once
         - `alpha`: the factor by which the jammer's power is attenuated
                at the receiver
         - `sigma_squared`: the noise variance (across all channels) at the 
                receiver 
         - `p_recv`: the power of the transmitted signal, at the receiver
         - `rates`: the rates at which the transmitter can communicate
        """
        self.k = k
        self.m = m
        self.p_avg = p_avg
        self.p_max = p_max
        self.c = c
        self.l = l
        self.n = n
        self.alpha = alpha
        self.sigma_squared = sigma_squared
        self.p_recv = p_recv
        self.rates = rates

        validate_param("simulation parameters", "(m == len(rates) - 1)", True, 
            m == len(rates) - 1)

        self.calculate_sinr_limits()
        self.calculate_p_jam()

    def calculate_sinr_limits(self):
        """
        Calculate and save the SINR limits (gamma_i) for each rate in 
        self.rates. Uses the Shannon-Hartley theorem with an arbitrary (?)
        bandwidth B = 10 MHz.

        Formula: C <= B * log2(1 + SINR)
        C/B <= log2(1 + SINR)
        2^(C/B) <= 1 + SINR
        SINR >= 2^(C/B) - 1

        Note, self.rates is in Mbps, hence multiplying by 10^6.
        """
        b = 10e6
        self.sinr_limits = [ 2 ** (c * 1e6 / b) - 1 for c in self.rates ]

    def calculate_p_jam(self):
        """
        Calculate and save the power levels of the jammer.
        """
        self.p_jam = [ ((self.p_recv / self.sinr_limits[self.m - i]) 
            - self.sigma_squared) / self.alpha for i in range(self.m + 1)]

    def __str__(self):
        return (
              f"### Simulation Parameters ### " + 
            f"\n - Number of channels: {self.k}" + 
            f"\n - Number of rates: {self.m + 1}" + 
            f"\n - Average power constraint of jammer: {self.p_avg}" +
            f"\n - Maximum power constraint of jammer: {self.p_max}" + 
            f"\n - Hopping cost: {self.c} Mbps" + 
            f"\n - Jamming cost: {self.l} Mbps" +
            f"\n - Jammer sweep/listen size: {self.n} channels" + 
            f"\n - alpha: {self.alpha}" + 
            f"\n - sigma_squared: {self.sigma_squared}" + 
            f"\n - Transmission power at receivier: {self.p_recv}" + 
            f"\n - Rates: {self.rates} Mbps" +
            f"\n" +
            f"\n### Calculated Parameters ###" + 
            f"\n - SINR limits: {self.sinr_limits}" + 
            f"\n - Jammer power levels: {self.p_jam}" +
            f"\n"
        )

    def __repr__(self):
        return (f"Parameters(k = {self.k}, m = {self.m}, p_avg = {self.p_avg}, "
            + f"p_max = {self.p_max}, c = {self.c}, l = {self.l}, "
            + f"n = {self.n}, alpha = {self.alpha}, " 
            + f"sigma_squared = {self.sigma_squared}, p_recv = {self.p_recv}, "
            + f"rates = {self.rates})"    
        )

def validate_transmit_strategy(parameters: Parameters, f: 'dict',
        state_space: 'list[str]', action_space: 'list[str]'):
    
    validate_param("state space", "size", math.floor(parameters.k 
        / parameters.m), len(state_space))
    
    validate_param("action space", "size", parameters.k)

    def transmit_validate(p_name: str, expected, actual):
        validate_param("transmit strategy", p_name, expected, actual)
    
    for state in state_space:
        action_p_sum = 0
        for action in action_space:
            p = f[state][action]
            transmit_validate(f"0 <= f[{state}][{action}] <= 1", 
                True, 0 <= p and p <= 1)
            action_p_sum += p
        transmit_validate(f"sum of f[{state}]", 1, action_p_sum)
        transmit_validate(f"number of actions in f[{state}]", len(action_space), 
            len(f[state]))

    transmit_validate("number of states in f", len(state_space), len(f))


def validate_jammer_strategy(parameters: Parameters, y: 'list[float]'):

    def jammer_validate(p_name: str, expected, actual):
        validate_param("jammer strategy", p_name, expected, actual)    

    jammer_validate("number of elements", parameters.m + 1, len(y))

    for i, yi in enumerate(y):
        jammer_validate(f"0 <= y[{i}] <= 1", True, 0 <= yi and yi <= 1)

    jammer_validate("sum of elements", 1, sum(y))
    jammer_validate("power constraint satisfied", True, np.dot(
        parameters.p_jam, y) <= parameters.p_avg)
