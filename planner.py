import sys
from mdp import MDP

# Validate arguments
assert len(sys.argv) == 3
assert sys.argv[2] in ['lp', 'hpi']

input_filepath = sys.argv[1]
algorithm = sys.argv[2]

mdp = MDP(input_filepath)
print(mdp.transition_function[0,])
