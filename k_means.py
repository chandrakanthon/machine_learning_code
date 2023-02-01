#Assignment on K-Means Clustering algorithm

# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data=pd.read_csv("Mall_Customers.csv")
print(data)
x=data.iloc[:,[3,4]].values
print(x)
#Training the K-means algorithm on the training dataset
from sklearn.cluster import KMeans
k=KMeans(n_clusters=5, init='k-means++', random_state= 42)
clusters=k.fit_predict(x)
print("number of clusters are \n",clusters)
#Visualizing the Clusters
mtp.scatter(x[clusters==0,0],x[clusters==0,1],s=100,c='blue',label='Cluster 1')
mtp.scatter(x[clusters==1,0],x[clusters==1,1],s=100,c='green',label='Cluster 2')
mtp.scatter(x[clusters==2,0],x[clusters==2,1],s=100,c='red',label='Cluster 3')
mtp.scatter(x[clusters==3,0],x[clusters==3,1],s=100,c='cyan',label='Cluster 4')
mtp.scatter(x[clusters==4,0],x[clusters==4,1],s=100,c='magenta',label='Cluster 5')
mtp.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
mtp.title('Clusters of customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()
