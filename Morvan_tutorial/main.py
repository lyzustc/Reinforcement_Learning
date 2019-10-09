import gym
from rl import DeepQNetwork
from rl.pgac import PolicyGradient
import matplotlib.pyplot as plt

def main():
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


if __name__ == "__main__":
    main()