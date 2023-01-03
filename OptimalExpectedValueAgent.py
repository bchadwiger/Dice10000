import math

from Agent import Agent
import AgentUtilities
import Rules

import queue
import numpy as np
from itertools import combinations_with_replacement
from itertools import product
import math
import matplotlib.pyplot as plt
import time


class OptimalExpectedValueAgent(Agent):
    def __init__(self, rules=Rules.Rules(), **kwargs):
        super().__init__(rules)

        self.lut_factorial = self.build_lut_factorial()
        self.list_lookup_tables_additional_potential_scores = self.build_potential_additional_score_lookup_tables()
        self.A, self.weighted_immediate_scores, self.weight_current_score = self.build_system_of_equations_parameters()
        self.expected_additional_scores_when_rethrowing_n_dice = self.solve_linear_system_of_equations()

    def compute_action(self, obs):
        """
        Generates an action for a given observation.

        :param obs: dict of observations. Example observation:
        dice_values [2 2 3 4 1 1] <class 'numpy.ndarray'>
        current_score 0 <class 'int'>
        current_min_collect_score 450 <class 'int'>
        players_scores [0 0] <class 'numpy.ndarray'>
        players_turn [ True False] <class 'numpy.ndarray'>

        :return: the action as 8-bit vector
        """

        self.update_observations(obs)
        face_counts = AgentUtilities.get_face_counts(self.dice_values, self.rules)

        n_remaining = np.sum(self.dice_values > 0)

        if n_remaining == self.number_dice and AgentUtilities.is_straight(face_counts):
            # take straight
            return np.array([1] * self.number_dice + [0, 0], dtype=bool)

        list_take_and_fuse = []
        action_priority_queue = []  # queue.PriorityQueue()

        n_multiples = np.sum(face_counts[1:] >= self.min_multiple)
        if n_multiples > 0:
            for val in range(1, self.face_high + 1):
                if face_counts[val] >= self.min_multiple:
                    take = np.zeros(self.number_dice, dtype=bool)
                    take[self.dice_values == val] = True
                    list_take_and_fuse.append(np.r_[take, False])
                    # n_remaining = np.sum(self.dice_values > 0) - np.sum(take)

        if 0 < face_counts[1] < self.min_multiple:
            for i in range(1, face_counts[1]+1):
                take = np.zeros(self.number_dice, dtype=bool)
                idxs = np.nonzero(self.dice_values == 1)[:i]
                take[idxs] = True
                list_take_and_fuse.append(np.r_[take, False])

        if 0 < face_counts[5] < self.min_multiple:
            for i in range(1, face_counts[5]+1):
                take = np.zeros(self.number_dice, dtype=bool)
                idxs = np.nonzero(self.dice_values == 5)[:i]
                take[idxs] = True
                list_take_and_fuse.append(np.r_[take, False])
                if i == 2:
                    list_take_and_fuse.append(np.r_[take, True])  # accounts for fusing

        if len(list_take_and_fuse) == 0:
            return np.zeros(self.number_dice+2, dtype=bool)
        else:
            take_combos = product([False, True], repeat=len(list_take_and_fuse))
            for take_i in take_combos:
                if not np.array(take_i).any():
                    continue

                take_and_fuse = np.zeros(self.number_dice + 1, dtype=bool)
                for i, c_i in enumerate(take_i):
                    if c_i:
                        take_and_fuse |= list_take_and_fuse[i]

                score_when_taking = self.get_score_when_taking(take_and_fuse[:self.number_dice])
                overall_score_when_taking = self.current_score + score_when_taking

                is_collectible = False
                if overall_score_when_taking >= self.current_min_collect_score:
                    is_collectible = True

                if not is_collectible:
                    action_priority_queue.append((-overall_score_when_taking, np.r_[take_and_fuse, False]))
                else:
                    expected_score_when_rethrowing = self.get_expected_score_when_rethrowing(overall_score_when_taking,
                                                                                             take_and_fuse)
                    if expected_score_when_rethrowing > overall_score_when_taking:
                        action_priority_queue.append((-expected_score_when_rethrowing, np.r_[take_and_fuse, False]))
                    else:
                        action_priority_queue.append((-overall_score_when_taking, np.r_[take_and_fuse, True]))

            expected_scores, actions = zip(*action_priority_queue)
            idx_max_expected_score = np.argmin(expected_scores)
            return actions[idx_max_expected_score]

    def build_lut_factorial(self):
        return {i: np.prod(np.arange(1, i+1)) for i in range(1, self.rules.number_dice+1)}

    def build_potential_additional_score_lookup_tables(self, debug=False):
        list_lookup_tables_additional_potential_scores = []
        for n_remaining in range(1, self.number_dice+1):
            lut_i = {dice_values: list(AgentUtilities.get_potential_score(dice_values, self.rules)) for dice_values in
                     combinations_with_replacement(list(range(1, self.rules.face_high+1)), n_remaining)}
            list_lookup_tables_additional_potential_scores.append(lut_i)

        for i, lut_i_remaining in enumerate(list_lookup_tables_additional_potential_scores):
            prob = 0
            for k, v in lut_i_remaining.items():
                # if v[0] > 0 or debug:
                prob_k = self.lut_factorial[len(k)] * \
                         np.prod(list(1./self.lut_factorial[np.sum(np.array(k) == j)] for j in set(k))) * \
                         1./self.rules.face_high**np.sum(len(k))
                if debug:
                    prob += prob_k
                list_lookup_tables_additional_potential_scores[i][k].append(prob_k)

            if debug:
                test_prob = math.isclose(prob, 1.0)
                # print(f'prob: {prob:.18f}, math.isclose(prob, 1.0): {test_prob}')
                assert test_prob

        return list_lookup_tables_additional_potential_scores

    def build_system_of_equations_parameters(self):
        A = np.zeros([self.number_dice, self.number_dice])
        weight_current_score = np.zeros([self.number_dice])
        weighted_immediate_scores = np.zeros([self.number_dice])

        for i, lut_i_remaining in enumerate(self.list_lookup_tables_additional_potential_scores):
            for k, v in lut_i_remaining.items():
                if v[0] == 0:
                    continue
                weighted_immediate_scores[i] += v[0] * v[2]
                weight_current_score[i] += v[2]
                sum_v0 = np.sum(v[1] == 0)
                j = sum_v0 if sum_v0 > 0 else self.number_dice
                A[i, j-1] += v[2]

        return A, weighted_immediate_scores, weight_current_score

    def solve_linear_system_of_equations(self):
        return np.linalg.solve(np.eye(self.number_dice) - self.A, self.weighted_immediate_scores)

    def get_score_when_taking(self, take):
        dice_values_tmp = self.dice_values.copy()
        dice_values_tmp[~take] = 0
        return AgentUtilities.get_potential_score(dice_values_tmp, self.rules)[0]

        # score += self.expected_additional_scores_when_rethrowing_n_dice[n_remaining - 1]
        # return score

    def get_expected_score_when_rethrowing(self, overall_score_when_taking, take_and_fuse):
        n_remaining = np.sum(self.dice_values > 0) - np.sum(take_and_fuse[:self.number_dice])
        if take_and_fuse[-1]:
            n_remaining += 1

        return self.weight_current_score[n_remaining-1] * overall_score_when_taking + \
               self.expected_additional_scores_when_rethrowing_n_dice[n_remaining-1]


if __name__ == '__main__':
    agent = OptimalExpectedValueAgent()
    # print(agent.lut_factorial)
    # print('==========================')
    # for lut in agent.list_lookup_tables_additional_potential_scores:
    #     for k_, v_ in lut.items():
    #         print(f'{k_}: {v_}')
    #     print()

    print(f'Coefficient matrix A = \n{agent.A}')
    print(f'det(I - A): {np.linalg.det(agent.A - np.eye(6))}')
    print('==========================')
    # solve (I - A)f = b
    f = np.linalg.solve(np.eye(6) - agent.A, agent.weighted_immediate_scores)
    print(f'f solving (I - A)f = b:\n{f}')

    print(agent.weight_current_score)
    print('==========================')
    print(agent.weighted_immediate_scores)
    print('==========================')

    N = 100
    errors = np.zeros([N,])
    y = np.zeros([6,])
    for i in range(N):
        y = agent.A @ y + agent.weighted_immediate_scores
        print(y)
        errors[i] = np.sum((y - f)**2)
        print(f'Error: {errors[i]}')

    plt.figure()
    plt.plot(np.log10(errors))
    plt.grid()
    plt.xlabel('#Recursions')
    plt.ylabel('log_10(MSE)')
    plt.show()

    # t0 = time.time()
    # # agent.build_potential_additional_score_lookup_tables()
    # list_luts = agent.list_lookup_tables_additional_potential_scores
    # t1 = time.time()
    # print(f'Elapsed time: {t1-t0} s')
    # # for dict_elem in list_luts:
    # #     for k, v in dict_elem.items():
    # #         print(k,v)
    # for dict_elem in list_luts:
    #     print(dict_elem)
    #
    # agent.build_lut_factorial()
    #
    # agent.build_expected_additional_scores()
    # print(agent.expected_additional_scores)
