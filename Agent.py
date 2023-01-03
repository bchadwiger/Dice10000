import Rules

import numpy as np


class Agent:
    def __init__(self, rules):

        if not rules:
            self.rules = Rules.Rules()
        else:
            self.rules = rules

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

    def update_observations(self, obs):
        self.dice_values = obs['dice_values']
        self.current_score = obs['current_score']
        self.current_min_collect_score = obs['current_min_collect_score']
        self.players_scores = obs['players_scores']
        self.players_turn = obs['players_turn']

