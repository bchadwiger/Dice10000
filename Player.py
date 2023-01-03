from EpsGreedyAgent import EpsGreedyAgent
from ScoreWeightedEpsGreedyAgent import ScoreWeightedEpsGreedyAgent
from OptimalExpectedValueAgent import OptimalExpectedValueAgent


implemented_agents = {
    'greedy': EpsGreedyAgent,
    'score_weighted': ScoreWeightedEpsGreedyAgent,
    'optimal_expected': OptimalExpectedValueAgent
}


class Player:
    def __init__(self, name, agent_type, rules, **kwargs):
        self.name = name
        self.agent = implemented_agents[agent_type](rules, **kwargs)

    def compute_action(self, obs):
        return self.agent.compute_action(obs)
