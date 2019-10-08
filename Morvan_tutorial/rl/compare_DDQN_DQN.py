import gym
from DQN import DeepQNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make("Pendulum-v0")
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope("Natural_DQN"):
    natural_DQN = DeepQNetwork(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        use_double_q=False,
        sess=sess
    )

with tf.variable_scope("Double_DQN"):
    double_DQN = DeepQNetwork(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        use_double_q=True,
        sess=sess
    )
sess.run(tf.global_variables_initializer())

def train(learner):
    total_steps = 0
    s = env.reset()
    
    while True:
        env.render()

        a = learner.choose_action(s)
        f_action = (a - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        s_new, reward, done, info = env.step(np.array([f_action]))

        reward /= 10

        learner.store_transition(s, a, reward, s_new)

        if total_steps > MEMORY_SIZE:
            learner.learn()

        if total_steps - MEMORY_SIZE > 20000:
            break

        s = s_new
        total_steps += 1

    return learner.q

q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()