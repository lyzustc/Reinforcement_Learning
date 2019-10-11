import numpy as np
import tensorflow as tf

class PPO(object):
    A_LR = 0.0001
    C_LR = 0.0002
    A_UPDATE_STEPS = 10
    C_UPDATE_STEPS = 10
    METHOD = [
        dict(name='kl_pen', kl_target=0.01, lam=0.5),
        dict(name='clip', epsilon=0.2),
    ][1]
    
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action

        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.n_state], 'state')
        
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.n_action], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if self.METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:
                self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.0-self.METHOD['epsilon'], 1.0+self.METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.n_action, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.n_action, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run([self.atrain_op, self.kl_mean], {self.tfs: s, self.tfadv: adv, self.tflam: self.METHOD['lam']})
                if kl > 4 * self.METHOD['kl_target']:
                    break
            if kl < self.METHOD['kl_target'] / 1.5:
                self.METHOD['lam'] /= 2
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)
        else:
            for _ in range(self.A_UPDATE_STEPS):
                self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})

        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

    def choose_action(self, s):
        s = s.reshape(1, -1)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        s = s.reshape(1, -1)
        return self.sess.run(self.v, {self.tfs: s})[0, 0]



if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    
    EP_MAX = 1000
    EP_LEN = 200
    GAMMA = 0.9
    BATCH = 32

    env = gym.make('Pendulum-v0').unwrapped
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]

    learner = PPO(n_state, n_action)
    
    all_ep_r = []
    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0

        for t in range(EP_LEN):
            # env.render()
            a = learner.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r+8)/8)
            s = s_
            ep_r += r

            if (t+1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = learner.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r).reshape(-1, 1)
                buffer_s, buffer_a, buffer_r = [], [], []
                learner.update(bs, ba, br)

        if ep==0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(0.9*all_ep_r[-1]+0.1*ep_r)

        print("Ep: {}, Ep_r: {}".format(ep, ep_r))

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()