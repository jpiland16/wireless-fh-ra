import random

class QTable:
    def __init__(self, states: 'list[str]', actions: 'list[str]'):
        self.values = { s: {a: 0 for a in actions} for s in states}
        
    def select_action(self, state: str, epsilon: float = 0):
        possible_actions = self.values[state]
        if random.random() < epsilon:
            # Explore
            actions = list(possible_actions)
            selected_action = actions[random.randint(0, 
                len(possible_actions) - 1)]
        else:
            # Exploit
            selected_action = max(possible_actions, 
                key = lambda k: possible_actions[k])
        return selected_action
            