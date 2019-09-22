import numpy as np
import pandas as pd
from .tabular import tabular

class sarsa_lambda(tabular):
    def __init__(self,  actions,  alpha=0.01, epsilon=0.9, gamma=0.9, lambda_=0.9):
        super(sarsa_lambda, self).__init__(actions, alpha, epsilon, gamma)
        self.lambda_ = lambda_
        self.eligibility_trace = self.table.copy()

    def learn(self):
        s, a, r, s_new = self.store_list.pop(0)
        _, a_new, _, _ = self.store_list[0]
        self.check_state_exist(s_new)
        q_old = self.table.loc[s, a]

        if s_new != 'terminal':
            q_new = r + self.gamma * self.table.loc[s_new, a_new]
        else:
            q_new = r
 
        # self.eligibility_trace.loc[s,a] += 1
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        self.table += self.alpha * (q_new - q_old) * self.eligibility_trace

        self.eligibility_trace *= self.gamma * self.lambda_

        self.memory_full = False

    def check_state_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state
                )
            )
            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state
                )
            )