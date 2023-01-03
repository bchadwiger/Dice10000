from Agent import Agent
import AgentUtilities
import Rules

import numpy as np
from itertools import combinations_with_replacement
from itertools import product
import math
import matplotlib.pyplot as plt


class OptimalExpectedValueAgent(Agent):
    def __init__(self, rules=Rules.Rules(), log_path=None, log_top_actions=None, **kwargs):
        super().__init__(rules)

        self.lut_factorial = self.build_lut_factorial()
        self.list_lookup_tables_additional_potential_scores = self.build_potential_additional_score_lookup_tables()
        self.A, self.weighted_immediate_scores, self.weight_current_score = \
            self.build_system_of_equations_parameters_for_greedy_rethrowing_policy()
        self.expected_additional_scores_when_rethrowing_n_dice = self.solve_linear_system_of_equations()
        self.log_path = log_path
        if self.log_path is not None:
            self.log = True
            self.logfile = open(self.log_path, "w")
        else:
            self.log = False
        self.log_top_actions = log_top_actions

    def compute_action(self, obs):
        """
        Computes the action with the highest expected score for a given observation.

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

        if self.log:
            self.logfile.write('==========================================\n\n')
            self.logfile.write(f'Current score: {self.current_score}\n')
            self.logfile.write('Players\' scores:\n')
            for ps in self.players_scores:
                self.logfile.write(f'    {ps}\n')
            self.logfile.write(f'Current minimal score for collecting: {self.current_min_collect_score}\n\n')
            self.log_dice_values()

        n_remaining_before_take = np.sum(self.dice_values > 0)

        if n_remaining_before_take == self.number_dice and AgentUtilities.is_straight(face_counts):
            # take straight
            return np.array([1] * self.number_dice + [0, 0], dtype=bool)

        list_take_and_fuse = []
        actions_dict = {}

        # TODO: refactor the following code block to be reused for TODO in self.build_system_of_equations_parameters()
        n_multiples = np.sum(face_counts[1:] >= self.min_multiple)
        if n_multiples > 0:
            for val in range(1, self.face_high + 1):
                if face_counts[val] >= self.min_multiple:
                    take = np.zeros(self.number_dice, dtype=bool)
                    take[self.dice_values == val] = True
                    list_take_and_fuse.append(np.r_[take, False])

        if 0 < face_counts[1] < self.min_multiple:
            for i in range(1, face_counts[1]+1):
                take = np.zeros(self.number_dice, dtype=bool)
                idxs = np.nonzero(self.dice_values == 1)[0][:i]
                take[idxs] = True
                list_take_and_fuse.append(np.r_[take, False])

        if 0 < face_counts[5] < self.min_multiple:
            for i in range(1, face_counts[5]+1):
                take = np.zeros(self.number_dice, dtype=bool)
                idxs = np.nonzero(self.dice_values == 5)[0][:i]
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

                n_remaining_take_i = self.get_n_remaining(take_and_fuse)

                is_collectible = False
                if n_remaining_take_i > 0 and overall_score_when_taking >= self.current_min_collect_score:
                    is_collectible = True

                if not is_collectible:
                    action = tuple(np.r_[take_and_fuse, False])
                    if action not in actions_dict:
                        actions_dict[action] = overall_score_when_taking + \
                                               self.get_expected_additional_score_for_n_dice_remaining(
                                                   n_remaining_take_i)
                else:
                    expected_score_when_rethrowing = self.get_expected_total_score_when_not_collecting(
                        overall_score_when_taking, n_remaining_take_i)

                    # add both actions for taking and continuing with corresponding (expected) score
                    action = tuple(np.r_[take_and_fuse, False])
                    if action not in actions_dict:
                        actions_dict[action] = expected_score_when_rethrowing

                    action = tuple(np.r_[take_and_fuse, True])
                    if action not in actions_dict:
                        actions_dict[action] = overall_score_when_taking

            actions_list = [(v, np.array(k)) for k, v in actions_dict.items()]

            expected_scores, actions = zip(*actions_list)
            idx_max_expected_score = np.argmax(expected_scores)

            if self.log:
                self.log_actions(actions, expected_scores)

            return actions[idx_max_expected_score]

    def log_actions(self, actions, expected_scores):
        expected_scores_with_idx = [(exp_score, i) for i, exp_score in enumerate(expected_scores)]
        expected_scores_with_idx_sorted = sorted(expected_scores_with_idx, reverse=True)
        _, idxs_sorted = zip(*expected_scores_with_idx_sorted)
        # self.logfile.write(f'Top {self.log_top_actions} actions by expected score\n')
        self.logfile.write(f'-------------------------\n')
        if self.log_top_actions is None:
            n_log_actions = len(actions)
        else:
            n_log_actions = np.minimum(self.log_top_actions, len(actions))
        for i in range(n_log_actions):
            for act in actions[idxs_sorted[i]][:self.number_dice]:
                if act:
                    self.logfile.write(' X ')
                else:
                    self.logfile.write('   ')
            if actions[idxs_sorted[i]][-2]:
                self.logfile.write(' Fu')
            else:
                self.logfile.write('   ')
            if actions[idxs_sorted[i]][-1]:
                self.logfile.write(' Cl')
            else:
                self.logfile.write('   ')
            self.logfile.write(f' ({expected_scores[idxs_sorted[i]]:.3f})\n')
        self.logfile.write('\n\n')
        self.logfile.flush()

    def log_dice_values(self):
        vis_str = ''
        for value in self.dice_values:
            if not value:
                vis_str += ' _ '
            else:
                vis_str += f' {value} '
        vis_str += '\n'
        self.logfile.write(vis_str)

    def build_lut_factorial(self):
        """
        Creates a lookup table with factorials of the first few natural numbers, i.e.,
        {1: 1, 2: 2, 3: 6, 4: 24, 5: 120, 6: 720}
        :return: dict with numbers as keys and factorial of a number as values
        """
        return {i: np.prod(np.arange(1, i+1)) for i in range(1, self.rules.number_dice+1)}

    def build_potential_additional_score_lookup_tables(self, debug=False):
        """
        Creates a lookup table with dice combinations for a given number of remaining dice and corresponding scores for
        a combination. I.e., returns
        [
            {
                (1,): [100, array([ True]), 0.16666666666666666],
                (2,): [0, array([False]), 0.16666666666666666],
                (3,): [0, array([False]), 0.16666666666666666],
                (4,): [0, array([False]), 0.16666666666666666],
                (5,): [50, array([ True]), 0.16666666666666666],
                (6,): [0, array([False]), 0.16666666666666666],
            },
            {
                (1, 1): [200, array([ True,  True]), 0.027777777777777776],
                (1, 2): [100, array([ True, False]), 0.05555555555555555],
                (1, 3): [100, array([ True, False]), 0.05555555555555555],
                (1, 4): [100, array([ True, False]), 0.05555555555555555],
                (1, 5): [150, array([ True,  True]), 0.05555555555555555],
                (1, 6): [100, array([ True, False]), 0.05555555555555555],
                (2, 2): [0, array([False, False]), 0.027777777777777776],
                (2, 3): [0, array([False, False]), 0.05555555555555555],
                (2, 4): [0, array([False, False]), 0.05555555555555555],
                (2, 5): [50, array([False,  True]), 0.05555555555555555],
                (2, 6): [0, array([False, False]), 0.05555555555555555],
                (3, 3): [0, array([False, False]), 0.027777777777777776]
                ...
            },
            ...
        ]

        :param debug:
        :return: a list of dicts with one list element per number of remaining. Each dict maps dice combos to scores
        """
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

    def build_system_of_equations_parameters_for_greedy_rethrowing_policy(self):
        """
        Creates the parameters for a linear system of equations to compute the expected scores.

        When n dice are remaining and rethrown, for n in {1,2,...,6}, for a given n all combinations in the (n-1)th list
        element in the output of self.build_potential_additional_score_lookup_tables() are possible, yielding the
        respective scores. E.g., when rethrowing 1 die, the possible outcomes, scores and probabilities are

            (1,): [100, 0.16666],
            (2,): [0,   0.16666],
            (3,): [0,   0.16666],
            (4,): [0,   0.16666],
            (5,): [50,  0.16666],
            (6,): [0,   0.16666].

        In case of throwing a 1 or a 5 (prob. 1/3), the round continues and all dice are taken, such that all dice are
        thrown again; in all other cases (prob 2/3), the current score is lost and it is the next player's turn.
        So, the expected score can be computed as

            1/6*(current_score + 100) + 1/6*(current_score + 50) + expected_additional_score(n_remaining=6),

        which is equal to

            2/6*current_score + 1/6*100 + 1/6*50 + 2/6*expected_additional_score(n_remaining=6).
             ^                  \------  -----/        \------------------  ------------------/
             |                         \/                             \/
        weight_current_score[0]        |                              |
                                       |                            f[5] (see text below)
                          weighted_immediate_scores[0]

        expected_additional_score captures the expected score obtained when continuing the next dice throwi with
        n_remaining dice.

        This can be extended to all other numbers of remaining dice, and hence is captured in vectors and a matrix of
        parameters with the number of rows equal to the total number of dice. The weights of current_score are captured
        in the variable weight_current_score, with weight_current_score[i] representing the weight for
        n_remaining = i+1. The weighted immediate score are captured in the vector weighted_immediate_scores, with
        the same mapping of indices. The expected additional scores when rethrowing with n dice are captured in the
        vector f (again with the same mapping indices), which must be computed and can then be used to determine the
        action with best expected score.

        For each n, depending on the combination of dice thrown and taken, a particular number of dice are remaining
        for the next throw. E.g., in the example above, when throwing a 1 or a 5, 6 dice are thrown in the next round,
        with an expected score of f[6-1] = f[5]. When throwing with two dice and obtaining e.g. (1,4), the 1 die is
        collected, and one die is remaining for the next throw, which is represented by f[1-1] = f[0]. The likelihood
        for this combo is obtained from the output of self.build_potential_additional_score_lookup_tables(), and
        captured in the matrix A. With these parameters, the vector f is then obtained as solution to the linear system
        of equations (computed subsequently in self.solve_linear_system_of_equations())

            f = Af + weighted_immediate_scores

        The parameters for A, weight_current_score and weighted_immediate_scores are computed in this function.

        :return: matrix A, vector weighted_immediate_scores, vector weight_current_score
        """
        A = np.zeros([self.number_dice, self.number_dice])
        weight_current_score = np.zeros([self.number_dice])
        weighted_immediate_scores = np.zeros([self.number_dice])

        for i, lut_i_remaining in enumerate(self.list_lookup_tables_additional_potential_scores):
            for k, v in lut_i_remaining.items():
                if v[0] == 0:
                    continue
                # TODO: take into account that not all dice that are possible to be taken are necessarily taken
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

    def get_n_remaining(self, take_and_fuse):
        n_remaining = np.sum(self.dice_values > 0) - np.sum(take_and_fuse[:self.number_dice])
        if take_and_fuse[-1]:
            n_remaining += 1
        return n_remaining

    def get_expected_additional_score_for_n_dice_remaining(self, n_remaining):
        return self.expected_additional_scores_when_rethrowing_n_dice[n_remaining-1]

    def get_expected_total_score_when_not_collecting(self, overall_score_when_taking, n_remaining):
        return self.weight_current_score[n_remaining-1] * overall_score_when_taking + \
               self.get_expected_additional_score_for_n_dice_remaining(n_remaining)


if __name__ == '__main__':
    agent = OptimalExpectedValueAgent()
    print(agent.lut_factorial)
    print('==========================')
    for lut in agent.list_lookup_tables_additional_potential_scores:
        for k_, v_ in lut.items():
            print(f'{k_}: {v_}')
        print()
    #
    # print(f'Coefficient matrix A = \n{agent.A}')
    # print(f'det(I - A): {np.linalg.det(agent.A - np.eye(6))}')
    # print('==========================')
    # # solve (I - A)f = b
    # f = np.linalg.solve(np.eye(6) - agent.A, agent.weighted_immediate_scores)
    # print(f'f solving (I - A)f = b:\n{f}')
    #
    # print(agent.weight_current_score)
    # print('==========================')
    # print(agent.weighted_immediate_scores)
    # print('==========================')
    #
    # N = 100
    # errors = np.zeros([N,])
    # y = np.zeros([6,])
    # for i in range(N):
    #     y = agent.A @ y + agent.weighted_immediate_scores
    #     print(y)
    #     errors[i] = np.sum((y - f)**2)
    #     print(f'Error: {errors[i]}')
    #
    # plt.figure()
    # plt.plot(np.log10(errors))
    # plt.grid()
    # plt.xlabel('#Recursions')
    # plt.ylabel('log_10(MSE)')
    # plt.show()
