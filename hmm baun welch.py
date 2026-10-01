import numpy as np
from hmmlearn.hmm import CategoricalHMM

# Initial probabilities
pi = np.array([0.5,0.3,0.2])

# Transition matrix
A = np.array([
    [0.6,0.3,0.1],
    [0.2,0.5,0.3],
    [0.3,0.2,0.5]
])

# Emission matrix
B = np.array([
    [0.6,0.3,0.1],
    [0.1,0.3,0.6],
    [0.7,0.2,0.1]
])

# Observation sequences
O  = [2,1,0,2]
O1 = [0,1,2,1]
O2 = [2,2,2,2]
O3 = [0,0,0,0]
O4 = [1,1,1,1]

model = CategoricalHMM(n_components=3,init_params="")

model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

def make_sequence(O):
    return np.array(O).reshape(-1,1)

print("Likelihood O  =",model.score(make_sequence(O)))
print("Likelihood O1 =",model.score(make_sequence(O1)))
print("Likelihood O2 =",model.score(make_sequence(O2)))
print("Likelihood O3 =",model.score(make_sequence(O3)))
print("Likelihood O4 =",model.score(make_sequence(O4)))

states = model.predict(make_sequence(O))

names = ["Walking","Running","Resting"]

print("\nMost Probable Hidden States:")
print([names[i] for i in states])

# Baum-Welch Training
model.fit(make_sequence(O))

print("\nLearned Initial Probabilities:")
print(model.startprob_)

print("\nLearned Transition Matrix:")
print(model.transmat_)

print("\nLearned Emission Matrix:")
print(model.emissionprob_)