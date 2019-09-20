import env
import rl

def main():
    demo_env = env.one_dim_walk(6)
    actions = demo_env.get_actions()
    learner = rl.q_learning(actions)

    MAX_EPISODES = 10
    for _ in range(MAX_EPISODES):
        s = demo_env.reset()

        while s != 'terminal':
            a = learner.choose_action(s)
            demo_env.render()
            s_new, r = demo_env.step(a)
            learner.store_transition(s, a, r, s_new)
            if learner.memory_full:
                learner.learn()

            s = s_new
        
        demo_env.render()


if __name__ == "__main__":
    main()