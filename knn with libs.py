import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

for k in [3,5,7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Xtr,ytr)

    pred = model.predict(Xt)

    print("K =",k)
    print("Actual   :",yt)
    print("Predicted:",pred)
    print("Accuracy :",np.mean(pred==yt))