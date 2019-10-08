import env
from rl.tabular import *
from rl import DeepQNetwork

def main():
    demo_env = env.Maze()
    actions = demo_env.get_actions()
    learner = DeepQNetwork(demo_env.n_actions, demo_env.n_features,
                           learning_rate=0.01,
                           reward_decay=0.9,
                           e_greedy=0.9,
                           replace_target_iter=200,
                           memory_size=2000)

    MAX_EPISODES = 300
    step = 0
    for _ in range(MAX_EPISODES):
        s = demo_env.reset()
        demo_env.render()

        done = False
        while not done:
            a = learner.choose_action(s)   
            s_new, r, done = demo_env.step(a)
            demo_env.render()

            learner.store_transition(s, a, r, s_new)
            if (step>200) and (step % 5 == 0):
                learner.learn()

            s = s_new
            step += 1

        demo_env.render()


if __name__ == "__main__":
    main()