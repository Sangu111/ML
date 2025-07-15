import pandas as pd

data=pd.read_csv("D:/MCA NMIT/MCA NMIT/give_me_credit.csv")
data=data.dropna()

x=data.drop("SeriousDlqin2yrs",axis=1)
y=data["SeriousDlqin2yrs"]
print(data.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=58)
print("splited")

from sklearn.svm import SVC
model1 = SVC(kernel='rbf',C=1.0)
model1.fit(x_train,y_train)

y_pred=model1.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score
print("ACCURACY ",accuracy_score(y_test,y_pred))
print("Classification report ", classification_report(y_test, y_pred))

correct = x_test[y_test == y_pred]
wrong = x_test[y_test != y_pred]

print("top 5 correct prediction",correct.head())
print("top 5 wrong prediction",wrong.head())