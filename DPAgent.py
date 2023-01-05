from Agent import Agent
import AgentUtilities
import Rules

import numpy as np
from itertools import combinations_with_replacement
from itertools import product
import math


class DPAgent(Agent):
    def __init__(self, rules=Rules.Rules(), **kwargs):
        super().__init__(rules)

