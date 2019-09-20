import numpy as np
import pandas as pd

class sarsa_lambda(object):
    alpha = 0.01
    epsilon = 0.9
    gamma = 0.9
    lambda_ = 0.9

    def __init__(self, actions):
        self.table = pd.DataFrame(
            columns=actions,
            dtype=np.float64
        )
        self.actions = actions
        self.store_list = []
        self.memory_full = False
        self.eligibility_trace = self.table.copy()

    def choose_action(self, state):
        self.check_state_exist(state)
        state_actions = self.table.iloc[state, :]
        if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):
            action = np.random.choice(self.actions)
        else:
            action = state_actions.idxmax()

        return action

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

    def store_transition(self, s, a, r, s_new):
        if self.memory_full == False:
            self.store_list.append((s,a,r,s_new))
        if len(self.store_list) > 1:
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
            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state
                )
            )