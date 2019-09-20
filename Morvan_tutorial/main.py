import env
import rl

def main():
    N_STATES = 6
    ACTIONS = ['left', 'right']
    demo_env = env.one_dim_walk(N_STATES)
    learner = rl.q_learning(N_STATES, ACTIONS)

    MAX_EPISODES = 10
    for _ in range(MAX_EPISODES):
        demo_env.reset()
        s = 0
        is_terminate = False

        while is_terminate == False:
            demo_env.render()
            a = learner.choose_action(s)
            s_new, r, is_terminate = demo_env.step(a)
            learner.store_transition(s, a, r)
            if learner.memory_full:
                learner.learn(s_new, r)

            s = s_new
        
        demo_env.render()


if __name__ == "__main__":
    main()