import numpy as np

pi = np.array([0.5,0.3,0.2])

A = np.array([
    [0.6,0.3,0.1],
    [0.2,0.5,0.3],
    [0.3,0.2,0.5]
])

B = np.array([
    [0.6,0.3,0.1],
    [0.1,0.3,0.6],
    [0.7,0.2,0.1]
])

# High, Medium, Low, High
O = [2,1,0,2]

T = len(O)
N = len(pi)

beta = np.zeros((T,N))

# Initialization
beta[T-1] = 1

# Recursion
for t in range(T-2,-1,-1):
    for i in range(N):
        beta[t][i] = sum(
            A[i][j]*B[j][O[t+1]]*beta[t+1][j]
            for j in range(N)
        )

# Likelihood
likelihood = sum(
    pi[i]*B[i][O[0]]*beta[0][i]
    for i in range(N)
)

print("Backward Probability Table:")
print(beta)

print("\nLikelihood =",likelihood)