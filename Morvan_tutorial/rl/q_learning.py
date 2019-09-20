import numpy as np
import pandas as pd
from .tabular import tabular

class q_learning(tabular):
    def __init__(self, actions, alpha=0.01, epsilon=0.9, gamma=0.9):
        super(q_learning, self).__init__(actions, alpha, epsilon, gamma)

    def learn(self):
        s, a, r, s_new = self.store_list.pop()
        self.check_state_exist(s_new)
        q_old = self.table.loc[s, a]

        if s_new != 'terminal':
            q_new = r + self.gamma * self.table.iloc[s_new, :].max()
        else:
            q_new = r

        self.table.loc[s, a] += self.alpha * (q_new - q_old)

        self.memory_full = False

    def store_transition(self, s, a, r, s_new):
        if self.memory_full == False:
            self.store_list.append((s,a,r,s_new))
        if len(self.store_list) > 0:
            self.memory_full = True