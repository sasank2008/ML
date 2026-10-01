import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df = pd.read_csv("data.csv")

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.2,random_state=42)

def show(name,p,t,depth,leaves):
    print("\n",name)
    print("Accuracy :",accuracy_score(yt,p))
    print("Precision:",precision_score(yt,p,average="weighted"))
    print("Recall   :",recall_score(yt,p,average="weighted"))
    print("F1 Score :",f1_score(yt,p,average="weighted"))
    print("Tree Depth:",depth)
    print("Leaf Nodes:",leaves)
    print("Training Time:",t)

# ID3
start = time.time()
id3 = DecisionTreeClassifier(criterion="entropy",random_state=42)
id3.fit(Xtr,ytr)
p1 = id3.predict(Xt)
show("ID3",p1,time.time()-start,id3.get_depth(),id3.get_n_leaves())

# CART
start = time.time()
cart = DecisionTreeClassifier(criterion="gini",random_state=42)
cart.fit(Xtr,ytr)
p2 = cart.predict(Xt)
show("CART",p2,time.time()-start,cart.get_depth(),cart.get_n_leaves())

# C4.5
def entropy(y):
    _,c = np.unique(y,return_counts=True)
    p = c/len(y)
    return -np.sum(p*np.log2(p+1e-10))

def c45(X,y):
    if len(np.unique(y)) == 1:
        return (None,y[0])

    best,bf,bt = -1,None,None

    for f in range(X.shape[1]):
        v = np.unique(X[:,f])

        for t in (v[:-1]+v[1:])/2:
            l = X[:,f] <= t
            r = ~l

            if not l.any() or not r.any():
                continue

            yl,yr = y[l],y[r]
            gain = entropy(y)-(len(yl)*entropy(yl)+len(yr)*entropy(yr))/len(y)

            a,b = len(yl)/len(y),len(yr)/len(y)
            si = -(a*np.log2(a+1e-10)+b*np.log2(b+1e-10))
            ratio = gain/si if si else 0

            if ratio > best:
                best,bf,bt = ratio,f,t

    if bf is None or best <= 0:
        return (None,np.bincount(y).argmax())

    l = X[:,bf] <= bt
    return (bf,bt,c45(X[l],y[l]),c45(X[~l],y[~l]))

def predict_c45(tree,x):
    if tree[0] is None:
        return tree[1]
    return predict_c45(tree[2],x) if x[tree[0]] <= tree[1] else predict_c45(tree[3],x)

def info(tree,d=0):
    if tree[0] is None:
        return d,1
    a,b = info(tree[2],d+1),info(tree[3],d+1)
    return max(a[0],b[0]),a[1]+b[1]

start = time.time()
tree = c45(Xtr,ytr)
p3 = np.array([predict_c45(tree,x) for x in Xt])
d,l = info(tree)

show("C4.5",p3,time.time()-start,d,l)