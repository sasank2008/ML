import csv
import numpy as np

with open("data.csv") as f:
    data = list(csv.reader(f))[1:]

X = np.array([[float(v) for v in r[:-1]] for r in data])
y = np.array([r[-1] for r in data])

# Encode labels
labels = np.unique(y)
y = np.array([np.where(labels==v)[0][0] for v in y])

# Scaling
X = (X-X.mean(0))/(X.std(0)+1e-8)

n = int(0.8*len(X))

Xtr,Xt = X[:n],X[n:]
ytr,yt = y[:n],y[n:]

def knn_predict(x,k):
    dist = np.sqrt(np.sum((Xtr-x)**2,axis=1))
    idx = np.argsort(dist)[:k]
    values = ytr[idx]
    return np.bincount(values).argmax()

for k in [3,5,7]:
    pred = np.array([knn_predict(x,k) for x in Xt])

    print("K =",k)
    print("Actual   :",yt)
    print("Predicted:",pred)
    print("Accuracy :",np.mean(pred==yt))