import pandas as pd
data = ("E:/ML LAB/forestfires.csv")
df = pd.read_csv(data)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

X = df.drop(columns=['area'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination = 0.05)
outlier_labels = lof.fit_predict(X_scaled)

df['outlier'] = outlier_labels

print("Number of outliers detected :", sum(outlier_labels == -1))
print(df[df['outlier'] == -1].head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.countplot(x='outlier', data=df)
plt.title("LOF outlier detection on forest fires dataset")
plt.xlabel("outlier (-1) vs inlier (1)")
plt.show()