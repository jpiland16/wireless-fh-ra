import random

class QTable:
    def __init__(self, states: 'list[str]', actions: 'list[str]'):
        self.values = { s: {a: 0 for a in actions} for s in states}
        self.epsilon = 0
        
    def select_action(self, state: str):
        possible_actions = self.values[state]
        if random.random() < self.epsilon:
            # Explore
            actions = list(possible_actions)
            selected_action = actions[random.randint(0, 
                len(possible_actions) - 1)]
        else:
            # Exploit
            selected_action = max(possible_actions, 
                key = lambda k: possible_actions[k])
        return selected_action

    def __len__(self):
        return len(self.values)

    def __getitem__(self, state):
        """
        Makes this QTable behave similar to a dict as required by Model.
        """
        best_action_probability = 1 - self.epsilon
        each_random_probability = self.epsilon / len(self.values[state])
        possible_actions = self.values[state]
        best_action = max(possible_actions, 
            key = lambda k: possible_actions[k])
        
        return {
            a: best_action_probability + each_random_probability 
            if a == best_action else each_random_probability
            for a in self.values[state]
        }

    def __str__(self):
        return str({state: self[state] for state in self.values})

    def __iter__(self):
        return iter(self.values)
