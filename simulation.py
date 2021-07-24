import random, math

from parameters import Parameters, get_default_parameters
from model import Model, validate_jammer_strategy, validate_transmit_strategy

class Simulation:
    def __init__(self, f: dict, y: 'list[float]', model: Model, 
            initial_state: str = "j", precision: int = -1, debug: bool = False):

        self.model = model
        params = model.params

        validate_transmit_strategy(model, f, precision)
        validate_jammer_strategy(model, y, precision)

        self.debug = debug
        self.params = params
        self.f = f
        self.y = y
        self.initial_state = initial_state

        self.reset()

    def reset(self):
        self.state = self.initial_state
        self.total_tx_reward = 0

        self.reset_pn_sequence()
        self.reset_jam_sequence()

        self.current_tx_channel = self.pn_sequence[0]
        self.current_tx_rate_index = len(self.params.rates) - 1

        self.jam_single_channel = False
        
    def reset_pn_sequence(self):
        self.current_pn_index = 0
        self.pn_sequence = [random.randint(0, self.params.k - 1)
            for _ in range(self.params.t)]

    def reset_jam_sequence(self):
        self.current_jam_index = 0
        self.jam_sequence = self.get_sweep_sequence()
        self.current_jammed_channels = self.jam_sequence[0]

    def get_next_pn_channel(self):
        self.current_pn_index += 1
        if self.current_pn_index >= len(self.pn_sequence):
            self.current_pn_index = 0
        return self.pn_sequence[self.current_pn_index]

    def get_sweep_sequence(self):
        random_channel_sequence = [i for i in range(self.params.k)]
        random.shuffle(random_channel_sequence)

        n = self.params.n

        return [[random_channel_sequence[i * n + j] for j in range(
            min(n, len(random_channel_sequence) - i * n)
        )]
            for i in range(math.ceil(self.params.k / n))]

    def play_turn(self):

        # Send/receive a message
        pn_index = self.current_pn_index
        channel = self.current_tx_channel
        rate_index = self.current_tx_rate_index
        jammer_power_index = random.choices([i for i in range(len(self.y))], 
            self.y)[0]
        jam_index = self.current_jam_index
        jammed_channels = self.current_jammed_channels
        single_jam = self.jam_single_channel

        message_was_jammed = (channel in jammed_channels and 
            jammer_power_index > self.params.m - rate_index) or (single_jam and 
                self.params.get_single_channel_attack_sinr(jammer_power_index) 
                <= self.params.sinr_limits[rate_index]
            )

        # Add reward (loss) for successful transmission (interception)
        if message_was_jammed:
            self.total_tx_reward -= self.params.l
        else:
            self.total_tx_reward += self.params.rates[rate_index]

        # Determine whether the jammer overheard an ACK or NACK
        jammer_overheard_something = channel in jammed_channels
        jammer_overheard_ack = (jammer_overheard_something and 
            not message_was_jammed)
        jammer_overheard_nack = (jammer_overheard_something and 
            message_was_jammed)
        
        # Compute the new state
        if self.state == "j":
            if message_was_jammed:
                pass
            else:
                self.state = "1"
        else:
            if message_was_jammed:
                self.state = "j"
            else:
                self.state = str(int(self.state) + 1)
        
        # Choose the next action
        action_p_dict = self.f[self.state]
        tx_action = random.choices(self.model.action_space, [action_p_dict[k] 
            for k in action_p_dict])[0]
        
        if tx_action[0] == "s":
            # Stay
            pass
        else:
            # Hop to a new channel
            self.state = "j"
            self.current_tx_channel = self.get_next_pn_channel()
            self.total_tx_reward -= self.params.c

        new_rate_index = int(tx_action[1:])
        self.current_tx_rate_index = new_rate_index

        if jammer_overheard_nack:
            self.reset_jam_sequence()
            self.jam_single_channel = False
        elif jammer_overheard_ack:
            self.jam_single_channel = True
            self.current_jammed_channels = [channel]
        else:
            self.current_jam_index += 1
            self.current_jammed_channels = \
                self.jam_sequence[self.current_jam_index]

        if self.debug:
            info = (f"""### Game turn information ###""" +
                    f"""\n - Transmitted on channel {channel} at rate """ + 
                    f"""#{rate_index + 1} = {self.params.rates[rate_index]}""" +
                    f""" Mbps"""
                    f"""\n - Jammer was on {jammed_channels} """ +
                    f"""and overheard {'ACK' if jammer_overheard_ack else (
                            'NACK' if jammer_overheard_nack else "nothing"
                    )}.""" +
                    f"""\n - Message was jammed: {message_was_jammed}""" +
                    f"""\n - Required JPI for non-single jam: {self.params.m
                        - rate_index + 1}"""
                    f"""\n - Single channel jam: {single_jam}""" + 
                        ((f"""\n - Minimum SINR for transmission: {
                        self.params.sinr_limits[rate_index]}""" + 
                            f"""\n - SINR at receiver: {
                                self.params.get_single_channel_attack_sinr(
                                    jammer_power_index) 
                            }""") if single_jam else "") +
                    f"""\n - Current PN index: {pn_index}""" + 
                    f"""\n - Current jam index: {jam_index}""" + 
                    f"""\n - Action taken: {tx_action}"""
                    f"""\n - Jammer power index: {jammer_power_index} """
                    f"""\n - New state: {self.state}"""
                    f"""\n\n"""
            )

            print(info)
            input()


    def run(self):
        """
        Play a game of the specified length and return the total 
        transmitter reward. Resets the simulation to the original state
        after run is complete.
        """
        game_time = 0
        while game_time < self.params.t:
            self.play_turn()
            game_time += 1

        reward = self.total_tx_reward
        self.reset() 

        return reward / self.params.t