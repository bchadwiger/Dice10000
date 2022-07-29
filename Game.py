import DiceWorld
import Player
import Rules
from EnvStatus import *

import art

import numpy as np


class Game:
    def __init__(self, players, rules=None):
        self.players = players
        self.number_players = len(players)
        self.scores = np.zeros(self.number_players, dtype=np.uint)

        if not rules:
            self.rules = Rules.Rules()
        else:
            self.rules = rules

        self.env = DiceWorld.DiceWorld(self.number_players, rules)
        self.done = False

        self.counter = 0

    def visualize_action(self, action):
        print('ACTION')
        print('Take dice: ', end='')
        for i, act_i in enumerate(action[:self.rules.number_dice]):
            if act_i:
                print(f' {i} ', end='')

        print(f', Fuse: ', end='')
        if action[-2]:
            print('YES, ', end='')
        else:
            print('NO, ', end='')

        print('Collect:', end='')
        if action[-1]:
            print('YES')
        else:
            print('NO')

    def visualize_obs(self, obs, info=None, done=False):
        if hasattr(info, 'status'):
            if info['status'] == envStatus.ACTION_INVALID:
                print(info['description'])
            elif info['status'] == envStatus.ADVANCE_PLAYER:
                print(info['description'])
            elif info['status'] == envStatus.SAME_PLAYER:
                print(f'Next move of player {np.argmax(obs.players_turn)}')

        dice_values = obs['dice_values']
        players_scores = obs['players_scores']
        players_turn = obs['players_turn']

        print('+------------------------------------------------+')
        print('Scores')
        for i, score in enumerate(players_scores):
            if players_turn[i]:
                print('>', end='')
            else:
                print(' ', end='')
            print(f"{self.players[i].name:>10}: {score:5d}")
        print()

        if not done:

            if info:
                if hasattr(info, 'intermediate_obs'):
                    for obs in info.intermediate_obs:
                        self.visualize_obs(obs, None)

            self.visualize_dice(dice_values)
            print()

    def visualize_dice(self, dice_values):
        for value in dice_values:
            if not value:
                print(' _ ', end='')
            else:
                print(f' {value} ', end='')
        print()

    def play(self):
        art.tprint('Dice 10000', font='random')

        obs = self.env.reset()
        self.visualize_obs(obs)
        while not self.done:
            self.counter += 1

            action = self.players[self.env.get_players_turn()].act(obs)

            self.visualize_action(action)

            obs, rew, done, info = self.env.step(action)

            self.visualize_obs(obs, info, done)

            self.done = done

        # self.visualize_obs(obs, info)
        print(f"Player {np.argmax(obs['players_scores'])} won the game")

if __name__ == '__main__':
    rules = Rules.Rules()

    players = [
        Player.Player('Player1', 'greedy', rules=rules, eps=0),
        Player.Player('Player2', 'greedy', rules=rules, eps=0.3)
    ]

    game = Game(players, rules)
    game.play()
