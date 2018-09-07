import numpy as np
from functools import reduce


class Agent:
    def __init__(self, mdp, algorithm):
        self.mdp = mdp
        self.algorithm = algorithm

        # initial value function = all zeros
        self.value_function = np.zeros((mdp.num_states,), dtype=float)
        # print(self.value_function)

        # initial policy = argmax of value function
        self.policy = self.get_policy_from_value_function()
        # print(self.policy)

    def solve(self):
        if self.algorithm == 'lp':
            pass
        elif self.algorithm == 'hpi':
            self.hpi()

    def hpi(self):
        T = self.mdp.transition_function
        R = self.mdp.reward_function
        P = self.policy
        gamma = self.mdp.gamma
        # policy evaluation using value iteration
        A = []
        b = []
        for s in range(self.mdp.num_states):
            coeff_of_s = []
            constant_of_s = 0
            for s_prime in range(self.mdp.num_states):
                constant_of_s += T[s, P[s], s_prime] * R[s, P[s], s_prime]
                if s == s_prime:
                    coeff_of_s.append(1 - gamma * T[s, P[s], s_prime])
                else:
                    coeff_of_s.append(- gamma * T[s, P[s], s_prime])
            A.append(coeff_of_s)
            b.append(constant_of_s)
        A = np.asarray(A)
        b = np.asarray(b)
        # reset value function
        self.value_function = np.linalg.solve(A, b)
        # get new policy
        new_policy = self.get_policy_from_value_function()

        if new_policy == self.policy:
            for s in range(self.mdp.num_states):
                print(self.value_function[s], '\t', self.policy[s])
            return
        else:
            self.policy = new_policy
            self.hpi()

    def get_policy_from_value_function(self):
        return [self.argmax_actions(s) for s in range(self.mdp.num_states)]

    def argmax_actions(self, s):
        T = self.mdp.transition_function
        R = self.mdp.reward_function
        gamma = self.mdp.gamma

        def get_disc_rewards_per_action(a):
            disc_rewards_per_a_s_prime = list(map(
                lambda s_prime: T[s, a, s_prime] * (R[s, a, s_prime] + gamma * self.value_function[s_prime]),
                list(range(self.mdp.num_states))))
            ans = reduce(lambda _a, _b: _a + _b, disc_rewards_per_a_s_prime)
            return ans

        disc_rewards_per_action = list(map(get_disc_rewards_per_action, list(range(self.mdp.num_actions))))
        argmax_action = np.argmax(disc_rewards_per_action)
        return argmax_action
