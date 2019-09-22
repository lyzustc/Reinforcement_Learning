import numpy as np
import pandas as pd
from .tabular import tabular

class sarsa(tabular):
    def __init__(self,  actions,  alpha=0.01, epsilon=0.9, gamma=0.9):
        super(sarsa, self).__init__(actions, alpha, epsilon, gamma)

    def learn(self):
        s, a, r, s_new = self.store_list.pop(0)
        _, a_new, _, _ = self.store_list[0]
        self.check_state_exist(s_new)
        q_old = self.table.loc[s, a]

        if s_new != 'terminal':
            q_new = r + self.gamma * self.table.loc[s_new, a_new]
        else:
            q_new = r

        self.table.loc[s, a] += self.alpha * (q_new - q_old)

        self.memory_full = False