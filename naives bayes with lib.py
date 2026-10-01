import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv("data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.factorize(X[col])[0]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

X = X.fillna(X.median())

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Actual :",y_test)
print("Predicted:",y_pred)

print("\nAccuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))