import numpy as np
import tensorflow as tf
from DQN import DQN

class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]



class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p() / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)



class PRDQN(DQN):
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
        super(PRDQN, self). __init__(
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
        self.memory = Memory(memory_size)

    def _build_op(self):
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)
            self.prloss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._pr_train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.prloss)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)

    def learn(self):
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        s, a = batch_memory[:, :self.n_features], batch_memory[:, self.n_features].astype(int)
        r, s_ = batch_memory[:, self.n_features+1], batch_memory[:, -self.n_features:]

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval], 
            feed_dict={self.s_: s_, self.s: s}
            )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = a
        reward = r

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, abs_errors, self.cost = self.sess.run(
            [self._pr_train_op, self.abs_errors, self.prloss], 
            feed_dict={self.s: s, self.q_target: q_target, self.ISWeights: ISWeights}
            )
        self.memory.batch_update(tree_idx, abs_errors)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1



if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(21)
    MEMORY_SIZE = 10000

    sess = tf.Session()
    with tf.variable_scope('natural_DQN'):
        RL_natural = DQN(
            n_actions=3, n_features=2, sess=sess,
            e_greedy_increment=0.00005, memory_size=MEMORY_SIZE,
            learning_rate=0.005, replace_target_iter=500
        )

    with tf.variable_scope('prioritized_replay_DQN'):
        RL_prio = PRDQN(
            n_actions=3, n_features=2, sess=sess,
            e_greedy_increment=0.00005, memory_size=MEMORY_SIZE,
            learning_rate=0.005, replace_target_iter=500
        )
    sess.run(tf.global_variables_initializer())

    def train(learner):
        total_steps = 0
        steps = []
        episodes = []

        for i_episode in range(20):
            s = env.reset()
            while True:
                a = learner.choose_action(s)
                s_, r, done, info = env.step(a)
                if done:
                    r = 10
                learner.store_transition(s, a, r, s_)

                if total_steps > MEMORY_SIZE:
                    learner.learn()

                if done:
                    print("episode {} finished.".format(i_episode))
                    steps.append(total_steps)
                    episodes.append(i_episode)
                    break
                s = s_
                total_steps += 1

        return np.vstack((episodes, steps))

    his_natural = train(RL_natural)
    his_prio = train(RL_prio)

    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural_DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='prioritized_replay_DQN')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()