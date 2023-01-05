from Agent import Agent
import AgentUtilities
import Rules

import tensorflow.keras as keras

class DeepQAgent(Agent):
    def __init__(self, rules=Rules.Rules(), n_hidden_dense_layers=4, n_hidden_units=128, activation='relu', **kwargs):
        super().__init__(rules)
        self.n_hidden_dense_layers = n_hidden_dense_layers
        self.n_hidden_units = n_hidden_units
        self.activation = activation
        self.net = self.build_network()

    def build_network(self):
        # face_counts for 0 ... number_dice, each with a max of face_high
        inp_dice = keras.layers.Input(shape=((self.number_dice+1) * self.face_high,))
        # current score, agent's total score, max total score of other players
        inp_scores = keras.layers.Input(shape=(3,))

        inputs = keras.layers.Concatenate()([inp_dice, inp_scores])
        x = inputs
        for _ in range(self.n_hidden_dense_layers):
            x = keras.layers.Dense(self.n_hidden_units, activation=self.activation)(x)
        output = keras.layers.Dense(self.number_dice+2, activation='sigmoid')(x)

        net = keras.models.Model([inp_dice, inp_scores], output)
        return net


if __name__ == '__main__':
    agent = DeepQAgent()