import DiceWorld
import Player
import Rules
from EnvStatus import *

import art

import numpy as np


class Game:
    def __init__(self, players, rules=None, render_mode='ascii'):
        self.players = players
        self.number_players = len(players)
        self.scores = np.zeros(self.number_players, dtype=np.uint)

        if not rules:
            self.rules = Rules.Rules()
        else:
            self.rules = rules

        self.render_mode = render_mode

        self.env = DiceWorld.DiceWorld([p.name for p in players], rules, render_mode=render_mode)
        self.done = False

        self.counter = 0

    # def visualize_obs(self, obs, info=None, done=False):
    #     if info:
    #         if 'intermediate_obs' in info and info['intermediate_obs']:
    #             N_intermediate = len(info['intermediate_obs'])
    #             for i, obs_ in enumerate(info['intermediate_obs']):
    #                 self.visualize_scores(obs_, None)
    #                 dice_values = obs_['dice_values']
    #                 self.visualize_dice(dice_values)
    #                 if N_intermediate > 1 and i < N_intermediate - 1:
    #                     print('No action possible. Advance players\' turn')
    #
    #         if 'description' in info:
    #             print(info['description'])
    #
    #     dice_values = obs['dice_values']
    #     self.visualize_scores(obs, info)
    #     self.visualize_dice(dice_values)
    #
    #     print()

    def read_and_parse_input(self):

        print('Which dice (0-based positions) should be taken?')
        while True:
            try:
                dice_to_take_raw = np.array(list(map(int, input().strip().split())))
                dice_to_take = self.encode_dice(dice_to_take_raw)

                if self.env.is_any_dice_to_take_already_taken(dice_to_take):
                    print('Cannot take a die that\'s already taken. Choose again!')
                    continue

                if not self.env.are_dice_to_take_valid(dice_to_take):
                    print('Some selected dice cannot be taken. Choose again!')
                    continue

                if not self.env.is_at_least_one_die_taken(dice_to_take):
                    print('At least one die must be taken. Choose again!')
                    continue

                break

            except ValueError():
                print('Invalid input. Try again')

        if self.env.is_fusable(dice_to_take):
            print('Fuse two 5s? (y/n)')
            while True:
                inp = input()
                if inp in ['y', 'Y']:
                    fuse = True
                    break
                if inp in ['n', 'N']:
                    fuse = False
                    break
        else:
            fuse = False

        if self.env.is_collectible(dice_to_take, fuse):
            print(f'Collect? (y/n) ({self.env.get_score(dice_to_take) + self.env.get_current_score()})')
            while True:
                try:
                    inp = input()
                    if inp in ['y', 'Y']:
                        collect = True
                        break
                    if inp in ['n', 'N']:
                        collect = False
                        break
                except ValueError():
                    print('Invalid input. Try again')
        else:
            collect = False

        return self.encode_action(dice_to_take_raw, fuse, collect)

    def encode_dice(self, dice_to_take):
        action_die = np.zeros(self.rules.number_dice, dtype=bool)
        for die_idx in dice_to_take:
            action_die[die_idx] = True
        return action_die

    def encode_action(self, dice_to_take, fuse, collect):
        if (dice_to_take > self.rules.number_dice).any():
            return np.zeros(self.rules.number_dice + 2, dtype=bool)  # default invalid action

        action = np.zeros(self.rules.number_dice+2, dtype=bool)

        action[:self.rules.number_dice] = self.encode_dice(dice_to_take)

        if fuse:
            action[self.rules.number_dice] = True
        if collect:
            action[self.rules.number_dice+1] = True
        return action

    def render(self, step_by_step=False):
        if self.render_mode == 'ascii':
            text = self.env.render(mode=self.render_mode)
            if not step_by_step:
                print(text, end='')
            else:
                for line in text.splitlines():
                    if line.startswith(self.env.str_no_action_possible):
                        print(line)
                        _ = input('Press any key to continue')
                    else:
                        print(line)
        else:
            raise NotImplementedError()

    def play(self, interactive=False, step_by_step=False):
        art.tprint('Dice 10000', font='random')

        obs, info = self.env.reset(return_info=True)
        self.render()

        while not self.done:
            self.counter += 1

            if interactive:
                if self.env.get_players_turn() == 0:
                    action = self.read_and_parse_input()
                else:
                    action = self.players[self.env.get_players_turn()].act(obs)
            else:
                action = self.players[self.env.get_players_turn()].act(obs)

            obs, rew, done, info = self.env.step(action)
            # if not done:
            self.render(step_by_step)

            if step_by_step:
                _ = input('Press any key to continue')
            self.done = done


if __name__ == '__main__':
    rules = Rules.Rules()

    players = [
        Player.Player('Player1', 'greedy', rules=rules, eps=0),
        Player.Player('Player2', 'greedy', rules=rules, eps=0.3)
    ]

    game = Game(players, rules)
    game.play(step_by_step=True, interactive=True)
