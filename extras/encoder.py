import sys

# Validate arguments
assert len(sys.argv) == 2

win_prob = float(sys.argv[1])
S = 101  # 0 - 100, 0 - lost, 100 - won
A = 99  # 1 - 99
gamma = 0.99
type = 'episodic'

print(S)
print(A)
# reward function
for s in range(0, S):
    for a in range(1, A + 1):
        for s_prime in range(0, S):
            if s_prime == 100:
                print(1, '', end='')
            else:
                print(0, '', end='')
        print('')
# transition function
for s in range(0, S):
    for a in range(1, A + 1):
        # print('s = %d, a = %d ' % (s, a), end='')
        for s_prime in range(0, S):
            # if credit level is 0, then game over
            # so any action will take it to 0 deterministically
            if s == 0:
                if s_prime == 0:
                    print(1, '', end='')
                elif s_prime != 0:
                    print(0, '', end='')
                continue

            # if 0 < credit < 100
            if 0 < s < 100:
                # invalid action, lost
                if a > min(s, 100 - s):
                    if s_prime == 0:
                        print(1, '', end='')
                    else:
                        print(0, '', end='')
                # valid action, in game
                else:
                    # case : won stake
                    if s + a == s_prime:
                        print(win_prob, '', end='')
                    # case : lost stake
                    elif s - a == s_prime:
                        print(1 - win_prob, '', end='')
                    # case : transition not possible with stake won or lost
                    else:
                        print(0, '', end='')
                continue

            # if credit is 100, then also game over
            # so any action will take it to 100 deterministically
            if s == 100:
                if s_prime == 100:
                    print(1, '', end='')
                elif s_prime != 100:
                    print(0, '', end='')
                continue

        print('')

print(gamma)
print(type)
