import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df = pd.read_csv("data.csv")

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.2,random_state=42)

def entropy(y):
    _,c = np.unique(y,return_counts=True)
    p = c/len(y)
    return -np.sum(p*np.log2(p+1e-10))

def gini(y):
    _,c = np.unique(y,return_counts=True)
    p = c/len(y)
    return 1-np.sum(p*p)

def best_split(X,y,method):
    best,bf,bt = -1,None,None
    parent = entropy(y) if method != "cart" else gini(y)

    for f in range(X.shape[1]):
        v = np.unique(X[:,f])

        for t in (v[:-1]+v[1:])/2:
            l = X[:,f] <= t
            r = ~l

            if not l.any() or not r.any():
                continue

            yl,yr = y[l],y[r]
            fun = entropy if method != "cart" else gini
            gain = parent-(len(yl)*fun(yl)+len(yr)*fun(yr))/len(y)

            if method == "c45":
                a,b = len(yl)/len(y),len(yr)/len(y)
                si = -(a*np.log2(a+1e-10)+b*np.log2(b+1e-10))
                gain = gain/si if si else 0

            if gain > best:
                best,bf,bt = gain,f,t

    return bf,bt,best

def build(X,y,method):
    if len(np.unique(y)) == 1:
        return (None,y[0])

    f,t,score = best_split(X,y,method)

    if f is None or score <= 0:
        return (None,np.bincount(y).argmax())

    l = X[:,f] <= t
    return (f,t,build(X[l],y[l],method),build(X[~l],y[~l],method))

def predict_one(tree,x):
    if tree[0] is None:
        return tree[1]
    return predict_one(tree[2],x) if x[tree[0]] <= tree[1] else predict_one(tree[3],x)

def predict(tree,X):
    return np.array([predict_one(tree,x) for x in X])

def info(tree,d=0):
    if tree[0] is None:
        return d,1
    a,b = info(tree[2],d+1),info(tree[3],d+1)
    return max(a[0],b[0]),a[1]+b[1]

def show(name,tree,p,t):
    d,l = info(tree)
    print("\n",name)
    print("Accuracy :",accuracy_score(yt,p))
    print("Precision:",precision_score(yt,p,average="weighted"))
    print("Recall   :",recall_score(yt,p,average="weighted"))
    print("F1 Score :",f1_score(yt,p,average="weighted"))
    print("Tree Depth:",d)
    print("Leaf Nodes:",l)
    print("Training Time:",t)

for name,method in [("ID3","id3"),("C4.5","c45"),("CART","cart")]:
    start = time.time()
    tree = build(Xtr,ytr,method)
    pred = predict(tree,Xt)
    show(name,tree,pred,time.time()-start)