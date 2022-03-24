import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% create data file

#class 1 
x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)

#class 2
x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)

#class 3
x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {"x":x, "y":y}

df = pd.DataFrame(dictionary)

plt.scatter(x1,y1, color = "red")
plt.scatter(x2,y2, color = "green")
plt.scatter(x3,y3, color = "blue")
plt.show()

#%% dendrogram

from scipy.cluster.hierarchy import linkage, dendrogram
merg = linkage(df, method = "ward")
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

#%% HC

from sklearn.cluster import AgglomerativeClustering
hierarticel_cluster = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster = hierarticel_cluster.fit_predict(df)

df["Label"] = cluster

plt.scatter(df.x[df.Label == 0], df.y[df.Label == 0], color = "red")
plt.scatter(df.x[df.Label == 1], df.y[df.Label == 1], color = "green")
plt.scatter(df.x[df.Label == 2], df.y[df.Label == 2], color = "blue")
plt.show()






