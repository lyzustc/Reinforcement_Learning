import numpy as np
import pandas as pd

class q_learning(object):
    alpha = 0.1
    epsilon = 0.9
    gamma = 0.9

    def __init__(self, n_states, actions):
        self.table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns=actions,
        )
        self.actions = actions
        self.store_list = []
        self.memory_full = False
        self.terminal_s = n_states - 1

    def choose_action(self, state):
        state_actions = self.table.iloc[state, :]
        if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):
            action = np.random.choice(self.actions)
        else:
            action = state_actions.idxmax()

        return action

    def learn(self, s_new, r):
        s, a, _ = self.store_list.pop()
        q_old = self.table.loc[s, a]

        if s_new != self.terminal_s:
            q_new = r + self.gamma * self.table.iloc[s_new, :].max()
        else:
            q_new = r

        self.table.loc[s, a] += self.alpha * (q_new - q_old)

        self.memory_full = False

    def store_transition(self, s, a, r):
        if self.memory_full == False:
            self.store_list.append((s,a,r))
        if len(self.store_list) > 0:
            self.memory_full = True