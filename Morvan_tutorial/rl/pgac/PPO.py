import numpy as np
import tensorflow as tf
import threading
import gym
import queue

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

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()
                self.sess.run(self.update_oldpi_op)
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :self.n_state], data[:, self.n_state: self.n_state+self.n_action], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                
                if self.METHOD['name'] == 'kl_pen':
                    for _ in range(self.A_UPDATE_STEPS):
                        _, kl = self.sess.run([self.atrain_op, self.kl_mean], {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})
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

                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

    def choose_action(self, s):
        s = s.reshape(1, -1)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        s = s.reshape(1, -1)
        return self.sess.run(self.v, {self.tfs: s})[0, 0]



class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        while not COORD.should_stop():
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0

            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()
                    buffer_s, buffer_a, buffer_r = [], [], []

                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1
                if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or t == EP_LEN - 1:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r).reshape(-1, 1)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()
                        UPDATE_EVENT.set()
                    
                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()
                        break

            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(0.9*GLOBAL_RUNNING_R[-1]+0.1*ep_r)
            GLOBAL_EP += 1
            print('EP {}, Worker {}, Ep_r: {}'.format(GLOBAL_EP, self.wid, ep_r))



class DPPO(object):
    def __init__(self):
        globals()['GAME'] = 'Pendulum-v0'
        self.env = gym.make(GAME).unwrapped
        globals()['GLOBAL_PPO'] = PPO(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        globals()['EP_MAX'] = 1000
        globals()['EP_LEN'] = 200
        globals()['GAMMA'] = 0.9
        globals()['MIN_BATCH_SIZE'] = 64
        globals()['N_WORKER'] = 4
        globals()['GLOBAL_UPDATE_COUNTER'], globals()['GLOBAL_EP'] = 0, 0
        globals()['GLOBAL_RUNNING_R'] = []
        globals()['COORD'] = tf.train.Coordinator()
        globals()['QUEUE'] = queue.Queue()
        globals()['UPDATE_EVENT'], globals()['ROLLING_EVENT'] = threading.Event(), threading.Event()

        UPDATE_EVENT.clear()
        ROLLING_EVENT.set()
        workers = [Worker(wid=i) for i in range(N_WORKER)]

        threads = []
        for worker in workers:
            t = threading.Thread(target=worker.work, args=())
            t.start()
            threads.append(t)

        threads.append(threading.Thread(target=GLOBAL_PPO.update))
        threads[-1].start()
        COORD.join(threads)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    agent = DPPO()

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

    env = gym.make(GAME)
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]