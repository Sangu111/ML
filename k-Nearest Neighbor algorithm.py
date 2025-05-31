import pandas as pd
data = pd.read_csv("E:/ML LAB/iris_naivebayes.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())
X = data.drop('target',axis=1)
y = data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, train_size= 0.8,
                                                    random_state=58)
print("Data split Successfully")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Model Trained Successfully")
train_accuracy = knn.score (X_train,y_train)
test_accuracy = knn.score(X_test,y_test)

print("Trained Accuracy",train_accuracy)
print("Testing Accuracy",test_accuracy)

y_pred = knn.predict(X_test)
correct_predictions= (y_pred == y_test)
wrong_predictions = (y_pred != y_test )

print("Correct Prediction")
print(X_test[correct_predictions])

print("Wrong Prediction")
print(X_test[wrong_predictions])