import tensorflow as tf
import numpy as np

class DQN(object):
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
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.store_q = store_q

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()

        self.sess = sess

    def _build_layers(self, s, c_names):
        n_l1 = 20
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            out = tf.matmul(l1, w2) + b2

        return out

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            self.q_eval = self._build_layers(self.s, ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES])

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = self._build_layers(self.s_, ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES])

        self._build_op()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t,e in zip(t_params, e_params)]

    def _build_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, s):
        s = s.reshape(1,-1)

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict = {self.s: s})
            action = np.argmax(action_value)
            if self.store_q:
                if not hasattr(self, 'q'):
                    self.q = []
                    self.running_q = 0
                self.running_q = self.running_q * 0.99 + 0.11 * np.max(action_value)
                self.q.append(self.running_q)

        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def sample(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        batch_memory = self.memory[sample_index, :]
        s, a = batch_memory[:, :self.n_features], batch_memory[:, self.n_features].astype(int)
        r, s_ = batch_memory[:, self.n_features+1], batch_memory[:, -self.n_features:]

        return s, a, r, s_

    def learn(self):
        s, a, r, s_ = self.sample()
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

        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: s, self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1