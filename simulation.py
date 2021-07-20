import math
from parameters import Parameters

def get_game_parameters():
    p_max = 2
    alpha = 1
    sigma_squared = 0.01
    p_recv = 1

    return Parameters(
        k = 4,
        m = 7,
        p_avg = 0.83 * p_max,
        p_max = p_max,
        c = 50,
        l = 25,
        n = 1, 
        alpha = alpha,
        sigma_squared = sigma_squared,
        p_recv = p_recv,
        rates = [6, 9, 12, 18, 24, 36, 48, 54],
    )

def get_state_space(parameters: Parameters):
    
    states = ["j"]
    for i in range(math.floor(parameters.k / parameters.m) ):
        states.append(str(i))
        
    return states
    
def get_action_space(parameters: Parameters):
    return [ "s" + str(rate) for rate in parameters.rates ] + \
        [ "h" + str(rate) for rate in parameters.rates]
    

def get_transition_probability_vector(parameters: Parameters,
        state: str, action: str, state_space: 'list[str]',
        jammer_power_index: int):
            
    probs = {k: 0 for k in state_space}
    
    # Equation 9
    if state == "j" and action[0] == "h":
       probs["j"] = parameters.n / (parameters.k - 1) \
           if jammer_power_index > parameters.m - int(action[1:]) else 0
       probs["1"] = 1 - probs["j"]
       
    # Equation 11 (same as #9) ???
    elif action[0] == "h":
        probs["j"] = parameters.n / (parameters.k - 1) \
           if jammer_power_index > parameters.m - int(action[1:]) else 0
        probs["1"] = 1 - probs["j"]
       
    # Equation 12   
    else:
        x = int(state)
        r = int(action[1:])
        sinr_single_attack = parameters.p_recv / ( parameters.alpha * 
            parameters.n * parameters.p_jam[jammer_power_index]
             + parameters.sigma_squared)
        
        p_discover_next = parameters.n / (parameters.k - parameters.n * x)
        p_single_channel_attack = parameters.m * x / parameters.k
        
        probs["j"] = p_discover_next + p_single_channel_attack \
            if x < parameters.k / parameters.m and \
            jammer_power_index > parameters.m - r else (
                p_single_channel_attack if x < parameters.k / parameters.m and
                sinr_single_attack < parameters.sinr_limits[r]
                else 0
            )
        probs[str(x + 1)] = 1 - probs["j"]
    
    