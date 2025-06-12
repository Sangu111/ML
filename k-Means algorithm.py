import pandas as pd
data = pd.read_csv("E:/ML LAB/Mall_Customers.csv")
print(data.head())

X = data[['Annual Income (k$)','Spending Score (1-100)']]

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

X_scaled = scalar.fit_transform(X)
print("Features Scaled")

from sklearn .cluster import KMeans
inertia = []
for k in range(1,11):
    model = KMeans(n_clusters = k,random_state = 42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

    kmeans = KMeans(n_clusters = 2,random_state = 42)
    data['Clusters'] = kmeans.fit_predict(X_scaled)

Centroids = scalar.inverse_transform(kmeans.cluster_centers_)
print("\n Centroids")
for i,c in enumerate(Centroids):
     print(f"Clusters{i} : Income = {c[0]},score = {c[1]}")

print("\n Clusters Counts:")
print(data['Clusters'].value_counts().sort_index())

import matplotlib.pyplot as plt
plt.scatter(X_scaled[:,0],X_scaled[:,1],c= data['Clusters'])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c= 'black',marker = 'X')
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()