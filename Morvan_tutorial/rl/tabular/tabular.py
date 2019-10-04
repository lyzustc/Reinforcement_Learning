import numpy as np
import pandas as pd

class tabular(object):
    def __init__(self, actions, alpha=0.01, epsilon=0.9, gamma=0.9):
        self.table = pd.DataFrame(
            columns=actions,
            dtype=np.float64
        )
        self.actions = actions
        self.store_list = []
        self.memory_size = 1
        self.memory_full = False
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_action(self, state):
        self.check_state_exist(state)
        state_actions = self.table.loc[state, :]
        
        if (np.random.uniform() > self.epsilon):
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)

        return action

    def learn(self):
        pass

    def store_transition(self, s, a, r, s_new):
        if self.memory_full == False:
            self.store_list.append((s,a,r,s_new))
        if len(self.store_list) > self.memory_size:
            self.memory_full = True

    def check_state_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state
                )
            )