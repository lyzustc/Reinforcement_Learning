import tensorflow as tf
import numpy as np

class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None,], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None,], name="action_value")

        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation.reshape(1,-1)})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.reshape(-1))
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in range(len(self.ep_rs)-1, 0, -1):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    
    RENDER = False
    DISPLAY_REWARD_THRESHOLD = 400

    demo_env = gym.make('CartPole-v0')
    demo_env = demo_env.unwrapped
    demo_env.seed(1)

    print(demo_env.action_space)
    print(demo_env.observation_space)
    print(demo_env.observation_space.high)
    print(demo_env.observation_space.low)

    learner = PolicyGradient(
        n_actions=demo_env.action_space.n,
        n_features=demo_env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99
    )

    MAX_EPISODES = 3000
    step = 0
    for i_episode in range(MAX_EPISODES):
        s = demo_env.reset()
        
        while True:
            if RENDER:
                demo_env.render()

            a = learner.choose_action(s)   
            s_new, r, done, info = demo_env.step(a)
            learner.store_transition(s, a, r)
            
            if done:
                ep_rs_sum = sum(learner.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print('episode:{}, reward:{}'.format(i_episode, int(running_reward)))

                vt = learner.learn()

                if i_episode == 0:
                    plt.plot(vt)
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            s = s_new
            step += 1

        demo_env.render()