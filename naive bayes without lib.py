import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv("data.csv")

X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.factorize(X[col])[0]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

X = X.fillna(X.median()).values
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

classes = np.unique(y_train)

priors = {}
means = {}
vars = {}

for c in classes:
    Xc = X_train[y_train == c]
    priors[c] = len(Xc)/len(X_train)
    means[c] = np.mean(Xc,axis=0)
    vars[c] = np.var(Xc,axis=0) + 1e-9

def predict(x):
    scores = {}

    for c in classes:
        likelihood = -0.5*np.sum(
            np.log(2*np.pi*vars[c]) +
            ((x-means[c])**2)/vars[c]
        )
        scores[c] = np.log(priors[c]) + likelihood

    return max(scores,key=scores.get)

y_pred = np.array([predict(x) for x in X_test])

print("Actual :",y_test)
print("Predicted:",y_pred)

print("\nAccuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))