import numpy as np
import tensorflow as tf
import gym
import multiprocessing
import threading

class ACNet(object):
    ENTROPY_BETA = 0.01

    def __init__(self, scope, sess, N_S, N_A, a_bound=None, globalAC=None):
        self.sess = sess
        if scope == 'Global_Net':
            with tf.variable_scope('Global_Net'):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope, N_A)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope, N_A)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * a_bound[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()
                    self.exp_v = self.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):
                    if a_bound is not None:
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0,1]), a_bound[0], a_bound[1])
                    else:
                        self.A = tf.squeeze(normal_dist.sample(1), axis=[0,1])

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    global OPT_A, OPT_C
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope, N_A):
        w_init = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')

        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        s = s.reshape(1, -1)
        return self.sess.run(self.A, {self.s: s})



class Worker(object):
    GAMMA = 0.9
    def __init__(self, name, sess, env_name, globalAC):
        self.env = gym.make(env_name).unwrapped
        self.name = name
        self.sess = sess
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.shape[0]
        a_bound = [self.env.action_space.low, self.env.action_space.high]

        self.AC = ACNet(name, sess, n_state, n_action, a_bound, globalAC)

    def work(self):
        global COORD, GLOBAL_EP, MAX_GLOBAL_EP, MAX_EP_STEP, UPDATE_GLOBAL_ITER, GLOBAL_RUNNING_R
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_.reshape(1,-1)})[0,0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print("{}, Ep: {}, Ep_r: {}".format(self.name, GLOBAL_EP, GLOBAL_RUNNING_R[-1]))
                    GLOBAL_EP += 1
                    break



class A3C(object):
    def __init__(self, env_name):
        self.n_workers = multiprocessing.cpu_count()
        self.lr_a = 0.0001
        self.lr_c = 0.001
        self.env = gym.make(env_name)
        self.sess = tf.Session()

        globals()['MAX_EP_STEP'] = 200
        globals()['MAX_GLOBAL_EP'] = 1000
        globals()['UPDATE_GLOBAL_ITER'] = 10
        globals()['GLOBAL_RUNNING_R'] = []
        globals()['GLOBAL_EP'] = 0

        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.shape[0]
        a_bound = [self.env.action_space.low, self.env.action_space.high]
        with tf.device('/cpu:0'):
            globals()['OPT_A'] = tf.train.RMSPropOptimizer(self.lr_a, name='RMSPropA')
            globals()['OPT_C'] = tf.train.RMSPropOptimizer(self.lr_c, name='RMSPropC')
            self.global_ac = ACNet('Global_Net', self.sess, n_state, n_action, a_bound)
            workers = []

            for i in range(self.n_workers):
                workers.append(Worker('W_{}'.format(i), self.sess, env_name, self.global_ac))

        globals()['COORD'] = tf.train.Coordinator()
        self.sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        globals()['COORD'].join(worker_threads)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    learner = A3C('Pendulum-v0')
    plt.plot(np.arange(len(globals()['GLOBAL_RUNNING_R'])), globals()['GLOBAL_RUNNING_R'])
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()