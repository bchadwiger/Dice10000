import numpy as np


def is_straight(face_counts):
    return (face_counts[1:] == 1).all()


def get_straight_score(rules):
    return rules.straight_score


def get_score_1():
    return 100


def get_score_5():
    return 50


def contains_multiple(face_counts, rules):
    return (face_counts[1:] >= rules.min_multiple).any()


def get_face_counts(dice_values, rules):
    face_counts = np.zeros([rules.face_high + 1], dtype=np.uint8)
    for val in dice_values:
        face_counts[val] += 1
    return face_counts


def get_multiple_score(face_value, count, rules):
    assert count >= rules.min_multiple
    if face_value == 1:
        return 1000 * 2 ** (count - rules.min_multiple)
    else:
        return 100 * face_value * 2 ** (count - rules.min_multiple)


def get_potential_score(dice_values, rules):
    face_counts = get_face_counts(dice_values, rules)

    # get the potential score in the current turn, i.e., ignoring all previous scores

    if (face_counts[1:] == 1).all():
        return get_straight_score(rules), np.ones([rules.number_dice], dtype=bool)

    potential_score = 0

    is_to_take = np.zeros_like(dice_values).astype(bool)
    if contains_multiple(face_counts, rules):
        for val, count in enumerate(face_counts[1:], start=1):
            if count >= rules.min_multiple:
                potential_score += get_multiple_score(val, count, rules)
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
