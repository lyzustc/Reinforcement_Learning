import env
from rl.tabular import *

def main():
    demo_env = env.Maze()
    actions = demo_env.get_actions()
    learner = q_learning(actions, alpha=0.1)

    MAX_EPISODES = 30
    for _ in range(MAX_EPISODES):
        s = demo_env.reset()
        demo_env.render()

        while s != 'terminal':
            a = learner.choose_action(s)   
            s_new, r = demo_env.step(a)
            demo_env.render()

            learner.store_transition(s, a, r, s_new)
            if learner.memory_full:
                learner.learn()

            s = s_new

        demo_env.render()


if __name__ == "__main__":
    main()