import numpy as np


class MDP:
    def __init__(self, input_filepath):
        with open(input_filepath) as input_file:
            raw_mdp = input_file.readlines()
        # num states & num actions
        self.num_states = int(raw_mdp[0].strip())
        self.num_actions = int(raw_mdp[1].strip())
        # reward function
        raw_reward_function = raw_mdp[2: 2 + self.num_states * self.num_actions]
        self.reward_function = np.zeros((self.num_states, self.num_actions, self.num_states,))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                r_s_a = raw_reward_function[s * self.num_actions + a].strip().split()
                self.reward_function[s, a] = np.asarray(r_s_a, dtype=float)
        # print(self.reward_function)
        # transition function
        raw_trans_function = raw_mdp[2 + self.num_states * self.num_actions: 2 + 2 * self.num_states * self.num_actions]
        self.transition_function = np.zeros((self.num_states, self.num_actions, self.num_states,))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                t_s_a = raw_trans_function[s * self.num_actions + a].strip().split()
                self.transition_function[s, a] = np.asarray(t_s_a, dtype=float)
        # print(self.transition_function)
        # gamma
        self.gamma = float(raw_mdp[2 + 2 * self.num_states * self.num_actions].strip())
        # print(self.gamma)
        # type
        self.type = raw_mdp[2 + 2 * self.num_states * self.num_actions + 1]
        # print(self.type)

    def __str__(self):
        return ' #states = %d, #actions = %d\n reward_function = \n%s\n transition_function = \n%s\n gamma = %f\n type = %s' % \
               (self.num_states, self.num_actions, str(self.reward_function), str(self.transition_function),
                self.gamma,
                self.type)
