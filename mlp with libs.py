import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xt = sc.transform(Xt)

model = MLPClassifier(hidden_layer_sizes=(10,),activation="relu",learning_rate_init=0.001,max_iter=1000,random_state=42)

model.fit(Xtr,ytr)

pred = model.predict(Xt)

print("Actual   :",list(yt))
print("Predicted:",list(pred))
print("Accuracy :",accuracy_score(yt,pred))

print("\nHidden Layers / Neurons Analysis")
for h in [(5,),(10,),(20,),(10,10)]:
    m = MLPClassifier(hidden_layer_sizes=h,activation="relu",learning_rate_init=0.001,max_iter=1000,random_state=42)
    m.fit(Xtr,ytr)
    print("Hidden Layers:",h,"Accuracy:",accuracy_score(yt,m.predict(Xt)))

print("\nLearning Rate Analysis")
for lr in [0.0001,0.001,0.1]:
    m = MLPClassifier(hidden_layer_sizes=(10,),activation="relu",learning_rate_init=lr,max_iter=1000,random_state=42)
    m.fit(Xtr,ytr)
    print("LR:",lr,"Accuracy:",accuracy_score(yt,m.predict(Xt)))

print("\nEpoch Analysis")
for ep in [100,500,1000]:
    m = MLPClassifier(hidden_layer_sizes=(10,),activation="relu",learning_rate_init=0.001,max_iter=ep,random_state=42)
    m.fit(Xtr,ytr)
    print("Epochs:",ep,"Accuracy:",accuracy_score(yt,m.predict(Xt)))

print("\nActivation Analysis")
for act in ["relu","tanh","logistic"]:
    m = MLPClassifier(hidden_layer_sizes=(10,),activation=act,learning_rate_init=0.001,max_iter=1000,random_state=42)
    m.fit(Xtr,ytr)
    print("Activation:",act,"Accuracy:",accuracy_score(yt,m.predict(Xt)))