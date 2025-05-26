import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv("E:/ML LAB/PlayTennis.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())

# Convert categorical features to numerical
for col in data.columns[:-1]:
    data[col] = data[col].astype('category')
    mapping_dict = dict(enumerate(data[col].cat.categories))
    print(f"\nOriginal categories for '{col}': {data[col].cat.categories}")
    data[col] = data[col].cat.codes
    print(f"Numerical codes for '{col}': {mapping_dict}")
print("\nCategorical to Numerical Conversion Successful")

# Separate features (X) and target (y)
target = 'Play Tennis'
X = data.drop(target, axis=1)
y = data[target]
print("\nTarget Conversion Successful")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=50)
model.fit(X_train, y_train)
print("\nModel Trained")

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
print(f"\nTrain accuracy is: {train_accuracy}")
test_accuracy = model.score(X_test, y_test)
print(f"Test accuracy is: {test_accuracy}")

# Make a prediction on a sample
sample = pd.DataFrame([[2, 1, 0, 1]], columns=X.columns)
prediction = model.predict(sample)
print("Prediction=",prediction)