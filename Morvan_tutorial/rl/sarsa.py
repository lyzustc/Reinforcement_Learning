import numpy as np
import pandas as pd

class sarsa(object):
    alpha = 0.01
    epsilon = 0.9
    gamma = 0.9

    def __init__(self, actions):
        self.table = pd.DataFrame(
            columns=actions,
            dtype=np.float64
        )
        self.actions = actions
        self.store_list = []
        self.memory_full = False

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

        self.table.loc[s, a] += self.alpha * (q_new - q_old)

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