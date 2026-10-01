import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

sc = StandardScaler()

Xtr = sc.fit_transform(Xtr)
Xt = sc.transform(Xt)

# Perceptron using library
lr = 0.01
epochs = 100

model = Perceptron(max_iter=epochs,eta0=lr,random_state=42)

model.fit(Xtr,ytr)

pred = model.predict(Xt)

print("Actual   :",list(yt))
print("Predicted:",list(pred))
print("Accuracy :",accuracy_score(yt,pred))

print("\nLearning Rate / Epoch Analysis")

for lr in [0.001,0.01,0.1]:

    m = Perceptron(max_iter=100,eta0=lr,random_state=42)

    m.fit(Xtr,ytr)

    print("LR:",lr,"Accuracy:",accuracy_score(yt,m.predict(Xt)))

for ep in [10,50,100]:

    m = Perceptron(max_iter=ep,eta0=0.01,random_state=42)

    m.fit(Xtr,ytr)

    print("Epochs:",ep,"Accuracy:",accuracy_score(yt,m.predict(Xt)))