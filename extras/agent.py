import numpy as np
from functools import reduce
from scipy import optimize


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
            self.lp()
        elif self.algorithm == 'hpi':
            self.hpi()

    def lp(self):
        T = self.mdp.transition_function
        R = self.mdp.reward_function
        gamma = self.mdp.gamma
        # A_ub
        A_ub = []
        # b_ub
        b_ub = []
        # objective function
        c = []
        # bounds
        bounds = []
        if gamma == 1 and self.mdp.type == 'episodic':
            for s in range(self.mdp.num_states - 1):
                c.append(1)
                bounds.append((None, None))
                for a in range(self.mdp.num_actions):
                    coeff_of_s = []
                    constant_of_s = []
                    for s_prime in range(self.mdp.num_states):
                        constant_of_s.append(-1 * T[s, a, s_prime] * R[s, a, s_prime])
                        if s_prime != self.mdp.num_states - 1:
                            if s == s_prime:
                                coeff_of_s.append(gamma * T[s, a, s_prime] - 1)
                            else:
                                coeff_of_s.append(gamma * T[s, a, s_prime])
                    A_ub.append(coeff_of_s)
                    b_ub.append(sum(constant_of_s))
            A_ub = np.asarray(A_ub)
            b_ub = np.asarray(b_ub)

            new_value_function = list(optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds).x)
            new_value_function.append(0)
            self.value_function = np.asarray(new_value_function, dtype=float)
            self.policy = self.get_policy_from_value_function()
            self.print_answer()

        else:
            for s in range(self.mdp.num_states):
                c.append(1)
                bounds.append((None, None))
                for a in range(self.mdp.num_actions):
                    coeff_of_s = []
                    constant_of_s = 0
                    for s_prime in range(self.mdp.num_states):
                        constant_of_s += -1 * T[s, a, s_prime] * R[s, a, s_prime]
                        if s == s_prime:
                            coeff_of_s.append(gamma * T[s, a, s_prime] - 1)
                        else:
                            coeff_of_s.append(gamma * T[s, a, s_prime])
                    A_ub.append(coeff_of_s)
                    b_ub.append(constant_of_s)
            A_ub = np.asarray(A_ub)
            b_ub = np.asarray(b_ub)

            self.value_function = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds).x
            self.policy = self.get_policy_from_value_function()
            self.print_answer()

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

        if np.linalg.det(A) == 0 and self.mdp.type == 'episodic' and gamma == 1:
            A = A[:len(A) - 1, :len(A[0]) - 1]
            b = b[:len(b) - 1]
            # reset value function
            # print(A)
            # print(b)
            new_value_function = list(np.linalg.solve(A, b))
            new_value_function.append(0)
            new_value_function = np.asarray(new_value_function)
            # print(new_value_function, self.value_function)
            try:
                np.testing.assert_almost_equal(self.value_function, new_value_function, decimal=10)
                self.print_answer()
                return
            except AssertionError:
                self.value_function = new_value_function
        else:
            # reset value function
            self.value_function = np.linalg.solve(A, b)

        # policy improvement using new value function
        new_policy = self.get_policy_from_value_function()

        if new_policy == self.policy:
            self.print_answer()
            return
        else:
            self.policy = new_policy
            self.hpi()

    def print_answer(self):
        for s in range(self.mdp.num_states):
            print(self.value_function[s], '\t', self.policy[s])

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
