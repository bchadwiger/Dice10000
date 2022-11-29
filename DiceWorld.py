"""
Author: Benjamin Hadwiger

This class represents an environment of the dice game Zehntausend (ten thousand) that can be used for playing by a human
player or as an environment for reinforcement learning

Created based on "Environment Creation",  https://www.gymlibrary.ml/content/environment_creation/

"""

import Rules
# from EnvStatus import *

import numpy as np
import gym
from gym import spaces
# import pygame
import copy

REWARD_NO_ACTION_POSSIBLE = -10
REWARD_INVALID_ACTION = -10
REWARD_WON = 10000


class DiceWorld(gym.Env):

    def __init__(self, player_names=('Player1', 'Player2'), rules=None, render_mode=None,
                 size=5, window_size=512, debug=False):

        metadata = {"render_modes": ["human", "ascii"], "render_fps": 0.2}

        self.__player_names = player_names
        self.__number_players = len(player_names)

        if not rules:
            self.__rules = Rules.Rules()
        else:
            self.__rules = rules

        self.__render_mode = render_mode
        self.__render_str = ""  # used to remember text for visualization

        self.__number_dice = rules.number_dice
        self.__max_score = rules.max_score
        self.__min_collect_score = rules.min_collect_score
        self.__face_high = rules.face_high
        self.__min_multiple = rules.min_multiple  # minimum number of equal-value dice that make a pairing ("Pasch")
        self.__straight_score = rules.straight_score  # score for a straight (dice: 1-2-3-4-5-6)

        self.size = size  # size of the square grid
        self.window_size = 512  # size of the PyGame window
        self.debug = debug

        self.__dice_values = None  # current values of dice
        self.__current_score = None  # score in current round
        self.__current_min_collect_score = None  # minimum required score to collect
        self.__players_scores = None  # captures overall scores of all players throughout game
        self.__players_turn = None  # captures whose player's turn it is

        self.__intermediate_obs = []  # used to remember observations where no action was possible
        self.__collected_score = 0  # used to remember last collected score

        # Observations are the values of the dice, whether or not they are already taken or not,
        # and the current score
        self.observation_space = spaces.Dict(
            {
                'values': spaces.Tuple([spaces.Discrete(self.__face_high+1)] * self.__number_dice),
                'current_score': spaces.Box(0, np.inf, [1]),
                'current_min_collect_score': spaces.Box(0, np.inf, [1]),
                'players_scores': spaces.Box(0, np.inf, [self.__number_players]),
                'players_turn': spaces.Discrete(self.__number_players)
            }
        )

        # Actions: take die i (one-hot encoded, with one bit for each of number_dice dice), fuse, collect,
        # i.e. in total number_dice + 2 possible actions
        self.action_space = spaces.MultiBinary(self.__number_dice+2)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_players_turn(self):
        return np.argmax(self.__players_turn)

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
        self.__dice_values[~self.__dice_taken()] = np.random.randint(1, self.__face_high + 1,
                                                                     int(np.sum(~self.__dice_taken())))

    def __reset_dice_values(self):
        self.__dice_values = np.random.randint(1, self.__face_high + 1, [self.__number_dice], dtype='uint8')

    def __reset_current_score(self):
        self.__current_score = 0

    def __reset_collected_score(self):
        self.__collected_score = 0

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

    def __reset_intermediate_obs(self):
        self.__intermediate_obs = []

    def __reset_for_next_players_turn(self):
        self.__reset_current_score()

        while True:
            self.__reset_dice_values()
            self.__advance_players_turn()

            if self.is_any_action_possible():
                break

            self.__visualize_observation()
            self.__visualize_no_action_possible()

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__reset_current_score()
        self.__reset_players_scores()
        self.__reset_players_turn()
        self.__reset_current_min_collect_score()
        self.__reset_dice_values()

        self.__reset_intermediate_obs()
        self.__reset_collected_score()

        if not self.is_any_action_possible():
            self.__reset_for_next_players_turn()

        observation = self.__get_obs()
        self.__visualize_observation()

        return (observation, None) if return_info else observation

    def __is_straight(self):
        if not self.__dice_taken().any():
            return np.array([np.sum(self.__dice_values == i) == 1 for i in range(1, self.__face_high)]).all()
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

    def is_fusable(self, dice_to_take):
        # check that exactly two 5s are in the dice that are not taken,
        # and that both of these should be taken in the current step
        idxs_dice_5 = self.__dice_values == 5
        return (np.sum(idxs_dice_5) == 2) and (np.sum(idxs_dice_5 & dice_to_take) == 2)

    def is_any_dice_to_take_already_taken(self, dice_to_take):
        return (dice_to_take & self.__dice_taken()).any()

    def is_at_least_one_die_taken(self, dice_to_take):
        return np.sum(dice_to_take) > 0

    def are_all_dice_taken_after_take(self, dice_to_take):
        if np.sum(dice_to_take | self.__dice_taken()) == self.__number_dice:
            return True

    def are_dice_to_take_valid(self, dice_to_take):
        if not self.is_at_least_one_die_taken(dice_to_take):
            return False

        for pos, take_die in enumerate(dice_to_take):
            if take_die:
                if not self.__is_1_or_5(pos):
                    if not (self.__is_part_of_multiple(pos) and self.__is_to_take_multiple(pos, dice_to_take)):
                        if not (self.__is_straight() and self.__is_to_take_straight(dice_to_take)):
                            return False
        return True

    def is_collectible(self, dice_to_take, fuse):
        if not self.is_score_sufficient_to_take(dice_to_take):
            return False
        else:
            if self.are_all_dice_taken_after_take(dice_to_take) and (not self.is_fusable() or not fuse):
                return False
            else:
                return True

    def is_score_sufficient_to_take(self, dice_to_take):
        if self.__current_score + self.get_score(dice_to_take) >= self.__current_min_collect_score:
            return True

    def is_action_valid(self, dice_to_take, fuse, collect):
        """
        Check whether an action is valid for the current state. For interpretation of action input, see function step.
        :param action:
        :return:
        """

        if self.is_any_dice_to_take_already_taken(dice_to_take):
            return False

        if not self.are_dice_to_take_valid(dice_to_take):
            return False

        if not self.is_at_least_one_die_taken(dice_to_take):
            return False

        fused = False
        if fuse:
            if not self.is_fusable(dice_to_take):
                return False
            fused = True

        # check that at least one die is taken and in this step, and not all dice are taken,
        # otherwise cannot collect points
        if collect and not self.is_collectible(dice_to_take, fused):
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
        self.__current_score += self.get_score(dice_to_take)

        # set taken dice to 0 indicating that the respective die is taken
        self.__dice_values[dice_to_take] = 0

    def get_score(self, dice_to_take):
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
        self.__collected_score = self.__current_score
        self.__players_scores[self.__players_turn] += self.__current_score
        self.__current_min_collect_score = self.__current_score + 50

    def is_any_action_possible(self):
        # Checks whether at least on die can be taken, otherwise the next player gets to play
        # Check for straight would be redundant, as a straight contains 1s and 5s, i.e. if straight, the check for 1 or
        # 5 below is true

        if (self.__dice_values == 1).any() or (self.__dice_values == 5).any():
            return True
        else:
            for pos, val in enumerate(self.__dice_values):
                if val in [0, 1, 5]:
                    continue  # does not need to be checked; [1,5] checked by if statement above and 0 cannot be taken
                if self.__is_part_of_multiple(pos):
                    return True

        return False

    def step(self, action):
        """
        Apply the action to the environment and compute the next state of the environment
        :param action: vector of one-hot encoded actions. Bit 0,...,i,...,self.number_dice-1: take die #i. Bit
            self.number_dice: fuse two 5s to one 1. self.number_dice+1: collect points (i.e., stop turn)
        :return:
        """

        self.__reset_intermediate_obs()

        assert isinstance(action, np.ndarray)
        assert action.dtype == bool

        action_dict = self.__decode_action(action)

        dice_to_take = action_dict['dice_to_take']
        fuse = action_dict['fuse']
        collect = action_dict['collect']

        self.__visualize_action(action, dice_to_take)
        if not self.is_action_valid(dice_to_take, fuse, collect):
            self.__reset_current_min_collect_score()
            self.__reset_for_next_players_turn()
            self.__visualize_observation()

            return self.__get_obs(), REWARD_INVALID_ACTION, False, None

        else:
            if fuse:
                self.__fuse(dice_to_take)

            self.__take_dice(dice_to_take)

            if collect:
                self.__collect()
                done = self.__players_scores[self.__players_turn] >= self.__max_score

                if done:
                    self.__visualize_win()
                    self.__visualize_scores(done)

                    return self.__get_obs(), REWARD_WON, done, None

                self.__reset_for_next_players_turn()
                current_obs = copy.deepcopy(self.__get_obs())
                current_score = self.__current_score

                reward = current_score

                self.__visualize_observation()

                return current_obs, reward, done, None

            else:
                if self.__dice_taken().all():
                    self.__reset_dice_values()
                else:
                    self.__roll_remaining_dice()

                if self.is_any_action_possible():
                    self.__visualize_observation()
                    return self.__get_obs(), 0, False, None
                else:
                    self.__visualize_observation()
                    self.__visualize_no_action_possible()
                    self.__reset_current_min_collect_score()

                    self.__reset_for_next_players_turn()

                    self.__visualize_observation()
                    return self.__get_obs(), REWARD_NO_ACTION_POSSIBLE, False, None

    def __visualize_dice(self, dice_values):
        vis_str = ''
        for value in dice_values:
            if not value:
                vis_str += ' _ '
            else:
                vis_str += f' {value} '
        vis_str += '\n'
        self.__render_str += vis_str

    def __visualize_scores(self, done=False):
        players_scores = self.__get_obs()['players_scores']
        players_turn = self.__get_obs()['players_turn']

        vis_str = ''
        vis_str += '+-------------------------------------------------------+\n'
        vis_str += 'Scores\n'
        for i, score in enumerate(players_scores):
            if players_turn[i] and not done:
                vis_str += '>'
            else:
                vis_str += ' '
            if players_turn[i]:
                vis_str += f"{self.__player_names[i]:>10}: {score:5d}"
                if not done:
                    vis_str += f" ({self.__current_score})\n"
                else:
                    vis_str += "\n"
            else:
                vis_str += f"{self.__player_names[i]:>10}: {score:5d} \n"
        vis_str += '\n'
        self.__render_str += vis_str

    def __visualize_action(self, action, dice_to_take):
        vis_str = ''
        current_player_name = self.__player_names[np.argmax(self.__players_turn)]
        vis_str += f'ACTION ({current_player_name})\n'
        vis_str += 'Take dice:'
        for i, act_i in enumerate(action[:self.__number_dice]):
            if act_i:
                vis_str += f'  {i}'

        vis_str += f', Fuse: '
        if action[-2]:
            vis_str += 'YES, '
        else:
            vis_str += 'NO, '

        vis_str += 'Collect:'

        if action[-1]:
            vis_str += ' YES'
        else:
            vis_str += ' NO'
        vis_str += f' ({self.__current_score + self.get_score(dice_to_take)})\n'

        # return vis_str
        self.__render_str += vis_str

    def __visualize_no_action_possible(self):
        self.__render_str += 'No action possible. Advance players\' turn\n'

    def __visualize_observation(self):
        self.__visualize_scores()
        dice_values = self.__get_obs()['dice_values']
        self.__visualize_dice(dice_values)
        self.__render_str += '\n'

    def __visualize_win(self):
        self.__render_str += '+-------------------------------------------------------+\n'
        self.__render_str += f"PLAYER {np.argmax(self.__players_turn)} WON THE GAME!\n"

    def render(self, mode='ascii'):
        """
        Visualize current state of the environment
        :param mode:
        :return:
        """
        if mode == 'rgb_array':
            # return np.array(...)  # return RGB frame suitable for video
            raise NotImplementedError()
        elif mode == 'human':
            # pop up a window and render
            raise NotImplementedError()
        elif mode == 'ascii':

            output_str = self.__render_str[:]
            self.__render_str = ''

            return output_str

        else:
            super(DiceWorld, self).render(mode=mode)  # just raise an exception

    # def render(self, mode="human"):
    #     if self.window is None and mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and mode == "human":
    #         self.clock = pygame.time.Clock()
    #
    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((0, 0, 0))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels
    #
    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )
    #
    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )
    #
    #     if mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()
    #
    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )


if __name__ == '__main__':
    diceWorld = DiceWorld(debug=True)
    obs = diceWorld.reset()
    print(type(obs))
    for k, v in obs.items():
        print(k, v, type(v))
    print(diceWorld.observation_space)
    