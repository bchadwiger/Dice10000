"""
Author: Benjamin Hadwiger

This class represents an environment of the dice game Zehntausend (ten thousand) that can be used for playing by a human
player or as an environment for reinforcement learning

Created based on "Environment Creation",  https://www.gymlibrary.ml/content/environment_creation/

"""

import numpy as np
import gym
from gym import spaces
import pygame

face_high = 6
REWARD_INVALID_ACTION = -10


class DiceWorld(gym.Env):

    def __init__(self, number_dice:np.uint8=6, number_players:np.uint8=2, max_score:np.uint=10000,
                 min_collect_score=450, straight_score=2000, min_multiple=3,
                 debug=False):

        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        self.debug = debug
        self.__number_dice = number_dice
        self.__number_players = number_players
        self.__max_score = max_score
        self.__min_collect_score = min_collect_score

        self.__min_multiple = min_multiple  # minimum number of equal-value dice that make a pairing ("Pasch")
        self.__straight_score = straight_score  # score for a straight (dice: 1-2-3-4-5-6)

        self.window = 512  # size of the PyGame window

        self.__dice_values = None  # current values of dice
        self.__current_score = None  # score in current round
        self.__current_min_collect_score = None  # minimum required score to collect
        self.__players_scores = None  # captures overall scores of all players throughout game
        self.__players_turn = None  # captures whose player's turn it is

        # Observations are the values of the dice, whether or not they are already taken or not,
        # and the current score
        self.observation_space = spaces.Dict(
            {
                'values': spaces.Tuple([spaces.Discrete(face_high+1)] * number_dice),
                'current_score': spaces.Box(0, np.inf, [1]),
                'current_min_collect_score': spaces.Box(0, np.inf, [1]),
                'players_scores': spaces.Box(0, np.inf, [number_players]),
                'players_turn': spaces.Discrete(number_players)
            }
        )

        # Actions: take die i (one-hot encoded, with one bit for each of number_dice dice), fuse, collect,
        # i.e. in total number_dice + 2 possible actions
        self.action_space = spaces.MultiBinary(number_dice+2)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # def set_values(self, values):
    #     if isinstance(values, list) or isinstance(values, tuple):
    #         assert len(values) == self.__number_dice
    #         self.__dice_values = np.array(values)
    #     elif isinstance(values, np.array):
    #         assert values.shape[0] == self.__number_dice
    #         self.__dice_values = values
    #     else:
    #         raise ValueError('Can only set __dice_values with list, tuple or np.array')
    #
    # def get_values(self):
    #     return self.__dice_values

    def __get_obs(self):
        return {
            'dice_values': self.__dice_values,
            'current_score': self.__current_score,
            'current_min_collect_score': self.__current_min_collect_score,
            'players_scores': self.__players_scores,
            'players_turn': self.__players_turn
        }

    def __dice_taken(self):
        return self.__dice_values == 0

    def __roll_remaining_dice(self):
        self.__dice_values[~self.__dice_taken()] = np.random.randint(1, face_high + 1, int(np.sum(~self.__dice_taken())))

    def __reset_dice_values(self):
        self.__dice_values = np.random.randint(1, face_high + 1, [self.__number_dice], dtype='uint8')

    def __reset_current_score(self):
        self.__current_score = 0

    def __reset_current_min_collect_score(self):
        self.__current_min_collect_score = self.__min_collect_score

    def __reset_players_scores(self):
        self.__players_scores = np.zeros([self.__number_players], dtype='uint')

    def __advance_players_turn(self):
        assert np.sum(self.__players_turn) == 1

        idx_turn = np.argmax(self.__players_turn)
        self.__players_turn[idx_turn] = 0
        idx_next_player = (idx_turn + 1) % self.__number_players
        self.__players_turn[idx_next_player] = 1

        assert np.sum(self.__players_turn) == 1

    def __reset_players_turn(self):
        self.__players_turn = np.zeros([self.__number_players], dtype=bool)
        self.__players_turn[0] = 1

    def __reset_for_next_players_turn(self):
        # self._reset_dice_taken()
        self.__reset_dice_values()
        self.__reset_current_score()
        self.__advance_players_turn()

        # self._roll_remaining_dice()

        # observation = self._get_obs()
        # info = None
        # return (observation, info) if return_info else observation

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self._reset_dice_taken()
        self.__reset_dice_values()
        self.__reset_current_score()
        self.__reset_players_scores()
        self.__reset_players_turn()
        self.__reset_current_min_collect_score()

        observation = self.__get_obs()
        info = None
        return (observation, info) if return_info else observation

    def __is_straight(self):
        if not self.__dice_taken().any():
            return np.array([np.sum(self.__dice_values == i) == 1 for i in range(1, face_high)]).all()
        else:
            return False

    def __is_to_take_straight(self, dice_to_take):
        return np.sum(dice_to_take) == self.__number_dice

    def __is_part_of_multiple(self, pos):
        assert 0 <= pos <= self.__number_dice - 1
        if not self.__dice_taken()[pos]:
            return np.sum(self.__dice_values == self.__dice_values[pos]) >= self.__min_multiple
        else:
            return False

    def __is_to_take_multiple(self, pos, dice_to_take):
        # check if at least self._min_multiple dice are taken, otherwise multiple is invalid
        assert 0 <= pos <= self.__number_dice - 1
        if not self.__dice_taken()[pos]:
            return np.sum(self.__dice_values[dice_to_take] == self.__dice_values[pos]) >= self.__min_multiple
        else:
            return False

    def __is_1_or_5(self, pos):
        assert 0 <= pos <= self.__number_dice - 1
        return (self.__dice_values[pos] == 1) or (self.__dice_values[pos] == 5)

    def __is_fusable(self, dice_to_take):
        # check that exactly two 5s are in the dice that are not taken,
        # and that both of these should be taken in the current step
        idxs_dice_5 = self.__dice_values == 5
        return (np.sum(idxs_dice_5) == 2) and (np.sum(idxs_dice_5 & dice_to_take) == 2)

    def __is_action_valid(self, action_dict):
        """
        Check whether an action is valid for the current state. For interpretation of action input, see function step.
        :param action:
        :return:
        """
        dice_to_take = action_dict['dice_to_take']
        fuse = action_dict['fuse']
        collect = action_dict['collect']

        # check if any dice that are already taken should be taken
        if (dice_to_take & self.__dice_taken()).any():
            if self.debug:
                print('Invalid action: Taken dice must not be taken again.')
            return False

        fused = False
        # if action contains fuse, check if it is allowed
        if fuse:
            if not self.__is_fusable(dice_to_take):
                if self.debug:
                    print('Invalid action: Cannot fuse if number of dice that are 5 is not equal two and if not both'
                          'of these dice are taken in this step.')
                return False
            fused = True

        # check that dice that should be taken are valid, i.e., each die is either 1, 5, or part of a "Pasch", i.e.,
        # triplet, quadruplet, ..., or part of a straight (1-2-3-...-face_high)
        for pos, take_die in enumerate(dice_to_take):
            if take_die:
                if not self.__is_1_or_5(pos):
                    if not (self.__is_part_of_multiple(pos) and self.__is_to_take_multiple(pos, dice_to_take)):
                        if not (self.__is_straight() and self.__is_to_take_straight(dice_to_take)):
                            if self.debug:
                                print('Invalid action: Cannot take die if it is not equal to 1 or 5, part of a '
                                      'triplet, quadruplet, etc., or a straight')
                            return False

        # check that at least one die is taken and in this step, and not all dice are taken,
        # otherwise cannot collect points
        if collect:
            # at least one die must be taken
            if np.sum(dice_to_take) < 1:
                if self.debug:
                    print('Invalid Action: At least one die must be taken per round.')
                return False

            # cannot collect points if all dice are taken
            if not fused and np.sum(dice_to_take | self.__dice_taken()) == self.__number_dice:
                if self.debug:
                    print('Invalid action: Cannot collect if all dice are taken. In this case the turn continues.')
                return False

            if not self.__current_score + self.__get_score(dice_to_take) >= self.__current_min_collect_score:
                if self.debug:
                    print('Invalid action: Cannot collect if current_score is not bigger than minimum_collect_score')
                return False

        return True

    def __decode_action(self, action):

        return {
            'dice_to_take': action[:self.__number_dice],
            'fuse': action[self.__number_dice],
            'collect': action[self.__number_dice + 1]
        }

    def __take_dice(self, dice_to_take):
        assert dice_to_take.shape[0] == self.__number_dice
        assert np.sum(dice_to_take) > 0

        # get score of dice that should be taken in this step
        self.__current_score += self.__get_score(dice_to_take)

        # set taken dice to 0 indicating that the respective die is taken
        self.__dice_values[dice_to_take] = 0

    def __get_score(self, dice_to_take):
        assert dice_to_take.shape[0] == self.__number_dice

        # first check if a special pattern is present
        if self.__is_straight() and self.__is_to_take_straight(dice_to_take):
            return self.__straight_score

        score = 0
        taken = np.zeros_like(dice_to_take)
        for pos, take_die in enumerate(dice_to_take):
            if taken[pos] or not take_die:
                continue

            if self.__is_1_or_5(pos) and not self.__is_part_of_multiple(pos):
                score += self.__get_score_single(self.__dice_values[pos])
                taken[pos] = 1
                continue

            if self.__is_part_of_multiple(pos) and self.__is_to_take_multiple(pos, dice_to_take):
                idxs_multiple_taken = dice_to_take & (self.__dice_values == self.__dice_values[pos])
                score += self.__get_score_multiple(idxs_multiple_taken)
                taken[idxs_multiple_taken] = 1

        return score

    def __get_score_single(self, die_value):
        if die_value == 1:
            return 100
        else:
            return 50

    def __get_score_multiple(self, idxs_multiple_taken):
        n_multiple_taken = np.sum(idxs_multiple_taken)
        assert n_multiple_taken >= self.__min_multiple

        if self.__dice_values[idxs_multiple_taken][0] == 1:
            return 1000 * 2 ** (n_multiple_taken - self.__min_multiple)
        else:
            return self.__dice_values[idxs_multiple_taken][0] * 100 * 2 ** (n_multiple_taken - self.__min_multiple)

    def __fuse(self, dice_to_take):
        # fuse the two 5s dice, i.e. change one die to 1 and only take this one (we use the first one w.l.o.g.)
        idxs_dice_5 = self.__dice_values == 5
        self.__dice_values[idxs_dice_5][0] = 1
        dice_to_take[idxs_dice_5][1] = 0

    def __collect(self):
        # self._current_reward = self._current_score
        self.__players_scores[self.__players_turn] += self.__current_score
        self.__current_min_collect_score = self.__current_score + 50

    def step(self, action):
        """
        Apply the action to the environment and compute the next state of the environment
        :param action: vector of one-hot encoded actions. Bit 0,...,i,...,self.number_dice-1: take die #i. Bit
            self.number_dice: fuse two 5s to one 1. self.number_dice+1: collect points (i.e., stop turn)
        :return:
        """

        assert isinstance(action, np.ndarray)
        assert action.dtype == bool

        action_dict = self.__decode_action(action)

        dice_to_take = action_dict['dice_to_take']
        fuse = action_dict['fuse']
        collect = action_dict['collect']

        if not self.__is_action_valid(action_dict):
            # self._current_reward = REWARD_INVALID_ACTION
            self.__reset_current_min_collect_score()
            self.__reset_for_next_players_turn()
            return self.__get_obs(), REWARD_INVALID_ACTION, False, None
        else:
            self.__take_dice(dice_to_take)

            if fuse:
                self.__fuse(dice_to_take)

            if collect:
                self.__collect()
                done = self.__players_scores[self.__players_turn] >= self.__max_score
                self.__reset_for_next_players_turn()
                return self.__get_obs(), self.__current_score, done, None
            else:
                if self.__dice_taken().all():
                    self.__reset_dice_values()
                else:
                    self.__roll_remaining_dice()

                return self.__get_obs(), 0, False, None


if __name__ == '__main__':
    diceWorld = DiceWorld(debug=True)
    obs = diceWorld.reset()
    for k, v in obs.items():
        print(k, v)
