
# coding: utf-8

# In[8]:

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[18]:

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])


# In[36]:

plt.scatter(X[: , 0], X[: , 1])
plt.title('Data Set')
plt.show()


# In[35]:

K = 2
kmeans = KMeans(n_clusters=K, random_state=0).fit(X)


# In[31]:

labels = kmeans.labels_


# In[32]:

kmeans.predict([[0, 0], [4, 4]])


# In[33]:

centroids = kmeans.cluster_centers_


# In[37]:

colors = ["g", "r", "c", "b", "y","gray", "brown", "tan", "gold", "peru", "tomato", "aqua"]

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c = colors[labels[i]], marker = 'o')
plt.scatter(centroids[:, 0], centroids[:, 1], c= 'k', marker = 'x', s = 50,  zorder = 10)
plt.title('K means clustering for for K = %d' % (K))
plt.show()


# In[45]:

data_set_1 = np.loadtxt('./class1.txt')
data_set_2 = np.loadtxt('./class2.txt')


# In[46]:

plt.scatter(data_set_1[: , 0], data_set_1[:, 1], s=50, c = 'c', marker = 'o', linewidths = 5, zorder = 10, alpha = 0.6, label='Class 1')
plt.scatter(data_set_2[: , 0], data_set_2[:, 1], s=50, c = 'r', marker = 'x', linewidths = 5, zorder = 10, alpha = 0.6, label='Class 2')

legend = plt.legend(loc='upper right', shadow=True)
plt.title("Synthetic Data")
plt.xlabel("First Feature X1")
plt.ylabel("Second Feature X2")
plt.show()


# In[48]:

def do_KMeans(X, K = 2):
    cluster_size = K
    clf = KMeans(n_clusters = cluster_size)
    clf.fit(X)

    # Gives the final cetnroids of each cluster
    centroids = clf.cluster_centers_

    # Label of each data-point
    labels = clf.labels_


    colors = 10*["g", "r", "c", "b", "y","gray", "brown", "tan", "gold", "peru", "tomato", "aqua"]
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], c = colors[labels[i]], marker = 'o')

    plt.scatter(centroids[:, 0], centroids[:, 1], c= 'k', marker = 'x', s = 50,  zorder = 10)
    plt.title('K means clustering for for K = %d' % (K))
    plt.show()


# In[51]:

do_KMeans(data_set_1, 10)
do_KMeans(data_set_2, 10)


# In[ ]:



