# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# For the train set
train.isna().head()
# For the test set
test.isna().head()

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)
"""
print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show(g)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show(grid)
"""
train = train.drop(
    ['Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Age'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin',
                  'Embarked', 'SibSp', 'Age'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm='auto')
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
       n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))

# 1. What are the relevant features of the Titanic dataset. Why are they relevant?
# The relavent fetaures are: "Pclass, Sex, Age, Fare, Parch, SibSp". They are relvant because what class an passanger was in determined if they would be picked or not. # Your age mattered because people of certain ages was more likely to survive. Sex because women and children was chosen more often. If you are a parent or sibling you # are also more likely to be picked because you might have a relation to someone that have already been picked.

# 2. Can you find a parameter configuration to get a validation score greater than 62% ?
# No, the values either became lower or the same with both different features and parameters

# 3. What are the advantages/disadvantages of K-Means clustering?
# The advantages with K-Means clustering is that it is realtvly easy to implement, its scales good with large data sets. It can easily adapt to new examples. Fast and efficient in term of computational cost. The results of it are easy to interpret.
#
# The disadvantages with k-means clustering is that the number of clusters are an input variable. Its sensitve to outliers. Has troubles with clustering data of different size and density. It can give different results depending on the order of the data.
# 4. How can you address the weaknesses?
# We can address it weaknesses by using it on datasets where you know the amount of clusters that is needed. We can remove outliers before we run the algorithm so that they dont corrupt the clusters. If the data has differnt size and density see to that you try to scale it in a certain intervall as all the other data in the dataset.
