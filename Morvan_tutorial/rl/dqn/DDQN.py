import tensorflow as tf
import numpy as np
from DQN import DQN

class DoubleDQN(DQN):
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
        super(DoubleDQN, self). __init__(
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
        
    def learn(self, s, a, r, s_):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: s_, self.s: s}
        )
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size)
        q_next_new = self.sess.run(self.q_eval, feed_dict={self.s: s_})
        max_actions = np.argmax(q_next_new, axis=1)
        q_target[batch_index, a] = r + self.gamma * q_next[batch_index, max_actions]

        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: s, self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1



if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Pendulum-v0")
    env.seed(1)
    MEMORY_SIZE = 3000
    ACTION_SPACE = 11

    sess = tf.Session()
    with tf.variable_scope("Natural_DQN"):
        natural_DQN = DQN(
            n_actions=ACTION_SPACE,
            n_features=env.observation_space.shape[0],
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            store_q = True
        )

    with tf.variable_scope("Double_DQN"):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE,
            n_features=env.observation_space.shape[0],
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            store_q=True
        )
    sess.run(tf.global_variables_initializer())

    def train(learner):
        total_steps = 0
        s = env.reset()
        
        while True:
            # env.render()

            a = learner.choose_action(s)
            f_action = (a - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
            s_new, reward, done, info = env.step(np.array([f_action]))

            reward /= 10

            learner.store_transition(s, a, reward, s_new)

            if total_steps > MEMORY_SIZE:
                learner.learn(*learner.sample())
                ep_r = 0

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