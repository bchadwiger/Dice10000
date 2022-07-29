import EpsGreedyAgent

implemented_agents = {
    'greedy': EpsGreedyAgent.EpsGreedyAgent,
}


class Player:
    def __init__(self, name, agent_type, rules, **kwargs):
        self.name = name
        self.agent = implemented_agents[agent_type](rules, **kwargs)

    def act(self, obs):
        return self.agent.act(obs)
