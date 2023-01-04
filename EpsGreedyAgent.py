from Agent import Agent
import AgentUtilities

import numpy as np


class EpsGreedyAgent(Agent):
    def __init__(self, rules, eps=0, **kwargs):
        super().__init__(rules)

        self.eps = eps

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
        face_counts = AgentUtilities.get_face_counts(self.dice_values)

        if AgentUtilities.is_straight(face_counts):
            # take straight
            return np.array([1] * self.number_dice + [0, 0], dtype=bool)

        take = np.zeros([self.number_dice], dtype=bool)
        if AgentUtilities.contains_multiple(face_counts):
            # take multiple
            for val, count in enumerate(face_counts[1:], start=1):
                if count >= self.min_multiple:
                    take[self.dice_values == val] = 1

        # take all 1s and 5s
        take[self.dice_values == 1] = 1
        take[self.dice_values == 5] = 1

        # collect if possible
        collect = (self.current_score + AgentUtilities.get_potential_score(self.dice_values)[0] >=
                   self.current_min_collect_score) and \
                  (np.sum(~self.dice_values.astype(bool)) + np.sum(take) < self.number_dice)

        if collect and np.random.rand() < self.eps:
            collect = False

        return np.concatenate([take, np.array([0]), np.array([collect])]).astype(bool)
