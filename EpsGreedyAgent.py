import Rules

import numpy as np


class EpsGreedyAgent:
    def __init__(self, rules, eps=0, **kwargs):
        self.eps = eps

        if not rules:
            rules = Rules.Rules()
        self.face_high = rules.face_high
        self.min_multiple = rules.min_multiple
        self.number_dice = rules.number_dice
        self.straight_score = rules.straight_score
        self.number_dice = rules.number_dice

        self.face_counts = np.zeros([self.face_high+1], dtype=np.uint8)

        self.dice_values = None
        self.current_score = 0
        self.current_min_collect_score = 0
        self.players_scores = None
        self.players_turn = None

        self.accumulated_reward = 0

    def __update_observations(self, obs):
        self.face_counts = np.zeros([self.face_high+1], dtype=np.uint8)

        self.dice_values = obs['dice_values']
        self.current_score = obs['current_score']
        self.current_min_collect_score = obs['current_min_collect_score']
        self.players_scores = obs['players_scores']
        self.players_turn = obs['players_turn']

        for val in self.dice_values:
            self.face_counts[val] += 1

    def __is_straight(self):
        return (self.face_counts[1:] == 1).all()

    def __contains_multiple(self):
        return (self.face_counts[1:] >= self.min_multiple).any()

    def __get_potential_score(self):
        # get the potential score in the current turn, i.e., ignoring all previous scores
        potential_score = 0

        is_taken = np.zeros([self.number_dice], dtype=bool)
        if self.__contains_multiple():
            for val, count in enumerate(self.face_counts[1:], start=1):
                if count >= self.min_multiple:
                    if val == 1:
                        potential_score += 1000 * 2**(count - self.min_multiple)
                    else:
                        potential_score += 100 * val * 2**(count - self.min_multiple)
                    is_taken[self.dice_values == val] = 1

        for idx, val in enumerate(self.dice_values):
            if not is_taken[idx]:
                if val == 1:
                    potential_score += 100
                elif val == 5:
                    potential_score += 50

        return potential_score

    def act(self, obs):
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

        self.__update_observations(obs)

        if self.__is_straight():
            # take straight
            return np.array([1] * self.number_dice + [0, 0], dtype=bool)

        take = np.zeros([self.number_dice], dtype=bool)
        if self.__contains_multiple():
            # take multiple
            for val, count in enumerate(self.face_counts[1:], start=1):
                if count >= self.min_multiple:
                    take[self.dice_values == val] = 1

        # take all 1s and 5s
        take[self.dice_values == 1] = 1
        take[self.dice_values == 5] = 1

        # collect if possible
        collect = (self.current_score + self.__get_potential_score() >= self.current_min_collect_score) and \
                  (np.sum(take) < self.number_dice)

        if collect and np.random.rand() < self.eps:
            collect = False

        return np.concatenate([take, np.array([0]), np.array([collect])]).astype(bool)
