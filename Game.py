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

        self.env = DiceWorld.DiceWorld([p.name for p in players], rules)
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
                dice_to_take = np.array(list(map(int, input("elements of array:-").strip().split())))
                break
            except:
                pass
            print('Invalid input. Try again')

        print('Fuse two 5s? (y/n)')
        while True:
            try:
                inp = input()
                if inp in ['y', 'Y']:
                    fuse = True
                    break
                if inp in ['n', 'N']:
                    fuse = False
                    break
            except:
                pass
            print('Invalid input. Try again')

        print('Collect? (y/n)')
        while True:
            try:
                inp = input()
                if inp in ['y', 'Y']:
                    collect = True
                    break
                if inp in ['n', 'N']:
                    collect = False
                    break
            except:
                pass
            print('Invalid input. Try again')

        return self.encode_action(dice_to_take, fuse, collect)

    def encode_action(self, dice_to_take, fuse, collect):
        if (dice_to_take > self.rules.number_dice).any():
            return np.zeros(self.rules.number_dice + 2, dtype=bool)  # default invalid action

        action = np.zeros(self.rules.number_dice+2, dtype=bool)
        for die in dice_to_take:
            action[die] = 1
        if fuse:
            action[self.rules.number_dice] = True
        if collect:
            action[self.rules.number_dice+1] = True
        return action

    def render(self):
        if self.render_mode == 'ascii':
            text = self.env.render(mode=self.render_mode)
            print(text, end='')
        else:
            raise NotImplementedError()

    def play(self, interactive=False, step_by_step=False):
        art.tprint('Dice 10000', font='random')

        obs, info = self.env.reset(return_info=True)
        self.render()

        while not self.done:
            self.counter += 1

            if interactive:
                if self.env.get_players_turn()[0]:
                    action = self.read_and_parse_input()
                else:
                    action = self.players[self.env.get_players_turn()].act(obs)
            elif step_by_step:
                _ = input('Press any key')
                action = self.players[self.env.get_players_turn()].act(obs)
            else:
                action = self.players[self.env.get_players_turn()].act(obs)

            obs, rew, done, info = self.env.step(action)
            self.render()

            self.done = done

        print(f"Player {np.argmax(obs['players_scores'])} won the game")


if __name__ == '__main__':
    rules = Rules.Rules()

    players = [
        Player.Player('Player1', 'greedy', rules=rules, eps=0),
        Player.Player('Player2', 'greedy', rules=rules, eps=0.3)
    ]

    game = Game(players, rules)
    game.play(step_by_step=False, interactive=False)
