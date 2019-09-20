import time

class one_dim_walk(object):
    ACTIONS = ['left', 'right']
    FRESH_TIME = 0.3
    
    def __init__(self, n_states):
        self.n_states = n_states
        self.env_list = None

    def step(self, action):
        assert self.env_list[self.state] == 'o'
        if action == 'right':
            if self.state == self.n_states - 2:
                self.state = self.n_states - 1
                self.terminate = True
                r = 1
            else:
                self.state = self.state + 1
                r = 0
        else:
            r = 0
            if self.state == 0:
                self.state = 0
            else:
                self.state = self.state - 1
        
        self.n_steps += 1

        return self.state, r, self.terminate
    
    def render(self):
        self.env_list = ['-']*(self.n_states-1) + ['g']
        self.env_list[self.state] = 'o'

        print('\r{}'.format(''.join(self.env_list)))
        if self.terminate:
            print('\r The total number of steps in this episode is {}.'.format(self.n_steps))
            time.sleep(2)
        else:
            time.sleep(self.FRESH_TIME)

    def reset(self, initial_s=0):
        self.n_steps = 0
        self.env_list = ['-']*(self.n_states-1) + ['g']
        self.env_list[0] = 'o'
        self.terminate = False
        self.state = initial_s


if __name__ == '__main__':
    env = one_dim_walk(6)
    env.reset()
    env.render()

    for _ in range(4):
        s, _, _ = env.step('right')
        env.render()
    for _ in range(4):
        s, _, _ = env.step('left')
        env.render()
    for _ in range(5):
        s, _, _ = env.step('right')
        env.render()