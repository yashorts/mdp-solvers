import sys

# Validate arguments
assert len(sys.argv) == 2

win_prob = float(sys.argv[1])
S = [i for i in range(0, 101)]
A = [i for i in range(1, 51)]
gamma = 1
_type = 'episodic'

print(len(S))
print(len(A))
# reward function
for s in S:
    for a in A:
        for s_prime in S:
            if s_prime == 100 and s != 0 and s != 100:
                print(1, '', end='')
            elif s_prime == 100 and s == 0:
                print(0, '', end='')
            elif s_prime == 100 and s == 100:
                print(0, '', end='')
            elif s_prime == 0:
                print(0, '', end='')
            else:
                print(0, '', end='')
        print('')
# transition function
for s in S:
    for a in A:
        transition_s_a = []
        # print('s = %d, a = %d ' % (s, a), end='')
        for s_prime in S:
            # if credit level is 0, send to 100
            if s == 0:
                if s_prime == 100:
                    transition_s_a.append(1)
                else:
                    transition_s_a.append(0)
                continue

            # if 0 < credit < 100
            if 0 < s < 100:
                # invalid action, lost
                if a > min(s, 100 - s):
                    if s_prime == 0:
                        transition_s_a.append(1)
                    else:
                        transition_s_a.append(0)
                # valid action, in game
                else:
                    # case : won stake
                    if s + a == s_prime:
                        transition_s_a.append(win_prob)
                    # case : lost stake
                    elif s - a == s_prime:
                        transition_s_a.append(1 - win_prob)
                    # case : transition not possible with stake won or lost
                    else:
                        transition_s_a.append(0)
                continue

            # if credit is 100, end state trapped
            if s == 100:
                if s_prime == 100:
                    transition_s_a.append(1)
                else:
                    transition_s_a.append(0)
                continue
        print(*transition_s_a, sep=' ')

print(gamma)
print(_type)
