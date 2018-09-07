import sys
from mdp import MDP
from agent import Agent

# Validate arguments
assert len(sys.argv) == 3
assert sys.argv[2] in ['lp', 'hpi']

input_filepath = sys.argv[1]
algorithm = sys.argv[2]

mdp = MDP(input_filepath)
agent = Agent(mdp, algorithm)
agent.solve()
