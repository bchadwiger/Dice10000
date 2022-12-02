import argparse
import DiceWorld
import Player
import Rules

import art

import numpy as np
import copy
import time

import matplotlib.pyplot as plt


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

        self.counter = 0

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
        if self.render_mode is None:
            return
        elif self.render_mode == 'ascii':
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
        if self.render_mode == 'ascii':
            art.tprint('Dice 10000', font='random')

        dict_obs = {i: [] for i in range(self.number_players)}
        dict_actions = {i: [] for i in range(self.number_players)}
        dict_rewards = {i: [] for i in range(self.number_players)}
        winner = None

        obs, info = self.env.reset(return_info=True)
        dict_obs[self.env.get_players_turn()].append(copy.deepcopy(obs))
        self.render()

        while True:
            self.counter += 1

            players_turn = self.env.get_players_turn()

            if interactive:
                if self.env.get_players_turn() == 0:
                    action = self.read_and_parse_input()
                else:
                    action = self.players[players_turn].compute_action(obs)
            else:
                action = self.players[players_turn].compute_action(obs)

            dict_actions[players_turn].append(copy.deepcopy(action))

            obs, rew, done, info = self.env.step(action)
            dict_rewards[players_turn].append(rew)
            dict_obs[players_turn].append(copy.deepcopy(obs))
            self.render(step_by_step)

            if done:
                winner = self.env.get_players_turn()
                scores = self.env.get_scores()
                assert (scores[winner] > scores[:winner]).all()
                assert (scores[winner] > scores[winner+1:]).all()
                break

            if step_by_step:
                _ = input('Press any key to continue')

        return winner#, dict_obs, dict_actions, dict_rewards


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Ga')

    rules = Rules.Rules()

    players = [
        Player.Player('Player1', 'greedy', rules=rules, eps=0),
        Player.Player('Player2', 'greedy', rules=rules, eps=0.3)
    ]

    game = Game(players, rules)
    game.play(step_by_step=False, interactive=False)
    