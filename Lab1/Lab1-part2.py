import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# The data
customer_data = pd.read_csv('shopping_data.csv')

X = customer_data.iloc[:, 3:5].values

# Dendogram
linked = linkage(X, 'single')
labelList = range(200)
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

# Agglomerative clustering
cluster = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
plt.show()

# 1 How many clusters do you have? Explain your answer
# We have 5 clusters from what you can see from the colors. There could be two more but they are to small to count as clusters. So 5 destinct clusters.

# 2 Plot the clusters to see how actually the data has been clustered.
# Plotted data and 5 clusters was appropriate

# 3 What can you conclude by looking at the plot?
# I would say that we can see 5 destinct groups with different purchasing behaviors.
# We see that people that have a high annual income are put into two groups one with a high spending score and one with a low spending score. The same goes for the people with a low annual income. The interesting group though is the one with mean annual income as they tend to have an an average spending socre across the whole group.
