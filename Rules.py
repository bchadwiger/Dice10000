class Rules:
    def __init__(self, number_dice=6, face_high=6, max_score=10000, min_collect_score=450, min_multiple=3,
                 straight_score=2000):

        self.number_dice = number_dice
        self.face_high = face_high
        self.max_score = max_score
        self.min_collect_score = min_collect_score
        self.min_multiple = min_multiple
        self.straight_score = straight_score
