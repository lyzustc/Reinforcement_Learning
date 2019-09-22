import numpy as np
import pandas as pd
from .tabular import tabular

class MC_epsilon(tabular):
    def __init__(self, actions, epsilon=0.9, gamma=0.9):
        super(MC_epsilon, self).__init__(actions, epsilon=epsilon, gamma=gamma)
        self.times_table = pd.DataFrame(columns=actions, dtype=np.int64)

    def learn(self):
        G = 0
        while(len(self.store_list) > 0):
            s, a, r, s_new = self.store_list.pop()
            G =  self.gamma * G + r
            self.times_table.loc[s, a] += 1
            self.table.loc[s, a] += 1.0 / self.times_table.loc[s, a] * (G - self.table.loc[s,a])

        self.memory_full = False

    def store_transition(self, s, a, r, s_new):
        if self.memory_full == False:
            self.store_list.append((s,a,r,s_new))
        if s_new == 'terminal':
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
            self.times_table = self.times_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state 
                )
            )