import math
from itertools import combinations_with_replacement

from Rules import Rules

import numpy as np

rules = Rules()
number_dice = rules.number_dice


def is_straight(face_counts):
    return (face_counts[1:] == 1).all()


def get_straight_score():
    return rules.straight_score


def get_score_1():
    return 100


def get_score_5():
    return 50


def contains_multiple(face_counts):
    return (face_counts[1:] >= rules.min_multiple).any()


def get_face_counts(dice_values):
    face_counts = np.zeros([rules.face_high + 1], dtype=np.uint8)
    for val in dice_values:
        face_counts[val] += 1
    return face_counts


def get_multiple_score(face_value, count):
    assert count >= rules.min_multiple
    if face_value == 1:
        return 1000 * 2 ** (count - rules.min_multiple)
    else:
        return 100 * face_value * 2 ** (count - rules.min_multiple)


def get_potential_score(dice_values):
    face_counts = get_face_counts(dice_values)

    # get the potential score in the current turn, i.e., ignoring all previous scores

    if (face_counts[1:] == 1).all():
        return get_straight_score(), np.ones([rules.number_dice], dtype=bool)

    potential_score = 0

    is_to_take = np.zeros_like(dice_values).astype(bool)
    if contains_multiple(face_counts):
        for val, count in enumerate(face_counts[1:], start=1):
            if count >= rules.min_multiple:
                potential_score += get_multiple_score(val, count)
                is_to_take[np.array(dice_values) == val] = 1

    for idx, val in enumerate(dice_values):
        if not is_to_take[idx]:
            if val == 1:
                potential_score += get_score_1()
                is_to_take[idx] = 1
            elif val == 5:
                potential_score += get_score_5()
                is_to_take[idx] = 1

    return potential_score, is_to_take


def build_lut_factorial():
    """
    Creates a lookup table with factorials of the first few natural numbers, i.e.,
    {1: 1, 2: 2, 3: 6, 4: 24, 5: 120, 6: 720}
    :return: dict with numbers as keys and factorial of a number as values
    """
    return {i: np.prod(np.arange(1, i+1)) for i in range(1, rules.number_dice + 1)}


lut_factorial = build_lut_factorial()


def build_potential_additional_score_lookup_tables(debug=False):
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
    for n_remaining in range(1, number_dice + 1):
        lut_i = {dice_values: list(get_potential_score(dice_values)) for dice_values in
                 combinations_with_replacement(list(range(1, rules.face_high + 1)), n_remaining)}
        list_lookup_tables_additional_potential_scores.append(lut_i)

    for i, lut_i_remaining in enumerate(list_lookup_tables_additional_potential_scores):
        prob = 0
        for k, v in lut_i_remaining.items():
            # if v[0] > 0 or debug:
            prob_k = lut_factorial[len(k)] * \
                     np.prod(list(1. / lut_factorial[np.sum(np.array(k) == j)] for j in set(k))) * \
                     1. / rules.face_high ** np.sum(len(k))
            if debug:
                prob += prob_k
            list_lookup_tables_additional_potential_scores[i][k].append(prob_k)

        if debug:
            test_prob = math.isclose(prob, 1.0)
            # print(f'prob: {prob:.18f}, math.isclose(prob, 1.0): {test_prob}')
            assert test_prob

    return list_lookup_tables_additional_potential_scores

list_lookup_tables_additional_potential_scores = build_potential_additional_score_lookup_tables()

