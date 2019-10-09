import numpy as np
import tensorflow as tf

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        pass
    
    def learn(self, s, a, td):
        pass

    def choose_action(self, s):
        pass


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        pass

    def learn(self, s, r, s_):
        return