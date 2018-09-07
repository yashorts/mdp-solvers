S = 101  # 0 - 100
A = 100  # 0 - 99
gamma = 1
type = 'episodic'

print(S)
print(A)

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            print(str(0), '', end='')

        print('')

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            print(str(0), '', end='')

        print('')

print(gamma)
print(type)
