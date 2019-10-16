import numpy as np
import tensorflow as tf
from DQN import DQN

class DuelDQN(DQN):
    def __init__(
        self,
        n_actions,
        n_features,
        sess,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        store_q = False
    ):
        super(DuelDQN, self). __init__(
            n_actions,
            n_features,
            sess,
            learning_rate,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
            store_q
        )

    def _build_layers(self, s, c_names):
        n_l1 = 20
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        with tf.variable_scope('Value'):
            w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
            self.V = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Advantage'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            self.A = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Q'):
            out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

        return out



if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    MEMORY_SIZE = 3000
    ACTION_SPACE = 25

    sess = tf.Session()
    with tf.variable_scope('natural'):
        natural_DQN = DQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, sess=sess, learning_rate=0.001, replace_target_iter=200)

    with tf.variable_scope('dueling'):
        dueling_DQN = DuelDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, sess=sess, learning_rate=0.001, replace_target_iter=200)

    sess.run(tf.global_variables_initializer())


    def train(RL):
        acc_r = [0]
        total_steps = 0
        observation = env.reset()
        while True:
            action = RL.choose_action(observation)

            f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(np.array([f_action]))

            reward /= 10      # normalize to a range of (-1, 0)
            acc_r.append(reward + acc_r[-1])  # accumulated reward

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if total_steps-MEMORY_SIZE > 15000:
                break

            observation = observation_
            total_steps += 1
        return acc_r

    r_natural = train(natural_DQN)
    r_dueling = train(dueling_DQN)

    plt.figure(1)
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()

    plt.show()