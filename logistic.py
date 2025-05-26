import pandas as pd
data=pd.read_csv("E:/ML LAB/iris.csv")
print(data.head())
print(data.isnull().sum())
print(data.info())
x=data.drop("target",axis=1)
y=data["target"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)
print("Features Preprocessed")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=(train_test_split(X_scaled, y, test_size=0.2, train_size=0.8, random_state=42))
print("Data Split successful")

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
print("Data model successful")

train_score=model.score(x_train,y_train)
print("model accuracy is",train_score)
import numpy as np
new_sample = np.array([[5.2,4.8,2.0,0.0]])
new_scaled=scaler.transform(new_sample)
prediction = model.predict(new_scaled)
print("target=", prediction[0])