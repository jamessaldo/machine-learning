from sklearn import preprocessing as pre
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans #hanya digunakan untuk elbow method
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import random

# import data air_bnb
df = pd.read_csv('air_bnb.csv')
# deleting coloumn that have many label
df = df.drop(['id', 'name', 'host_name', 'host_id', 'latitude', 'longitude', 'last_review'],axis=1)
# labeling data
encode = pre.LabelEncoder()
df['room_type'] = encode.fit_transform(df['room_type'])
df['neighbourhood_group'] = encode.fit_transform(df['neighbourhood_group'])
df['neighbourhood'] = encode.fit_transform(df['neighbourhood'])
# Fillin NaN data with mean datas
df = df.fillna(df.mean())
# Normalize datas
norm = Normalizer()
df = pd.DataFrame(norm.fit_transform(df))
df.columns = ['neighbourhood_group', 'neighbourhood', 'room_type', 'price', 'minimum_nights', 
            'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
# Check the outlier datas
z = np.abs(stats.zscore(df))
df = df[(z < 3).all(axis=1)]
# feature engineering
sns.set_style("whitegrid")
cor = df.corr()
plt.figure(figsize=(9, 7))
ax = sns.heatmap(cor, vmin=-1, vmax=1, annot=True, fmt='.1g')
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.show()
# split datas from df and zip them to datas
dx, dy, dz = df['room_type'], df['price'], df['minimum_nights']
datas = list(zip(dx, dy, dz))
datas1 = datas.copy()
datas2 = datas.copy()
#elbow method untuk mencari nilai K untuk K-means 
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(np.array(datas))
    kmeanModel.fit(np.array(datas))
    distortions.append(sum(np.min(cdist(np.array(datas), kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / np.array(datas).shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# First model with K=3
# define first centroid
centroid1 = random.choice(datas2.copy())
centroid2 = random.choice([x for x in datas2.copy() if x != centroid1])
centroid3 = random.choice([x for x in datas2.copy() if x != centroid1 and x != centroid2])
# first plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dx, dy, dz, color='black', alpha=0.009)
ax.text2D(0.05, 0.95, 'First model with K=3', transform=ax.transAxes)
ax.scatter(centroid1[0], centroid1[1], centroid1[2], color='red')
ax.scatter(centroid2[0], centroid2[1], centroid2[2], color='green')
ax.scatter(centroid3[0], centroid3[1], centroid3[2], color='blue')
ax.set_xlabel('Room Type')
ax.set_ylabel('Price')
ax.set_zlabel('Minimum Nights')
plt.tight_layout()
plt.pause(1)
while True:
    clusredx, clusgrnx, clusblux = [], [], []
    clusredy, clusgrny, clusbluy = [], [], []
    clusredz, clusgrnz, clusbluz = [], [], []
    cluster1 = []
    # clustering points
    for data in datas1.copy():
        cal1 = (data[0]-centroid1[0])**2 + (data[1] - centroid1[1])**2 + (data[2] - centroid1[2])**2
        cal2 = (data[0]-centroid2[0])**2 + (data[1] - centroid2[1])**2 + (data[2] - centroid2[2])**2
        cal3 = (data[0]-centroid3[0])**2 + (data[1] - centroid3[1])**2 + (data[2] - centroid3[2])**2
        if min(cal1, cal2, cal3) == cal1:
            clusredx.append(data[0])
            clusredy.append(data[1])
            clusredz.append(data[2])
            cluster1.append(0)
        elif min(cal1, cal2, cal3) == cal2:
            clusgrnx.append(data[0])
            clusgrny.append(data[1])
            clusgrnz.append(data[2])
            cluster1.append(1)
        elif min(cal1, cal2, cal3) == cal3:
            clusblux.append(data[0])
            clusbluy.append(data[1])
            clusbluz.append(data[2])
            cluster1.append(2)
    # define new centroid
    a, b, c = centroid1, centroid2, centroid3
    centroid1 = [sum(clusredx)/len(clusredx), sum(clusredy)/len(clusredy), sum(clusredz)/len(clusredz)]
    centroid2 = [sum(clusgrnx)/len(clusgrnx), sum(clusgrny)/len(clusgrny), sum(clusgrnz)/len(clusgrnz)]
    centroid3 = [sum(clusblux)/len(clusblux), sum(clusbluy)/len(clusbluy), sum(clusbluz)/len(clusbluz)]
    # break condition
    if (a == centroid1) and (b == centroid2) and (c == centroid3):
        #calculate SSE for clustering
        sse1, sse2, sse3 = 0, 0, 0
        for i in range(len(clusredx)):
            sse1 = sse1 + ((clusredx[i]-centroid1[0])**2 + (clusredy[i] - centroid1[1])**2 + (clusredz[i] - centroid1[2])**2)
        for i in range(len(clusgrnx)):
            sse2 = sse2 + ((clusgrnx[i]-centroid2[0])**2 + (clusgrny[i] - centroid2[1])**2 + (clusgrnz[i] - centroid2[2])**2)
        for i in range(len(clusblux)):
            sse3 = sse3 + ((clusblux[i]-centroid3[0])**2 + (clusbluy[i] - centroid3[1])**2 + (clusbluz[i] - centroid3[2])**2)
        Sum1SSE = sse1 + sse2 + sse3
        break
# define final plot
plt.cla()
ax.text2D(0.05, 0.95, 'First model with K=3', transform=ax.transAxes)
ax.scatter(clusredx, clusredy, clusredz, color='red')
ax.scatter(clusgrnx, clusgrny, clusgrnz, color='green')
ax.scatter(clusblux, clusbluy, clusbluz, color='blue')
ax.set_xlabel('Room Type')
ax.set_ylabel('Price')
ax.set_zlabel('Minimum Nights')
plt.tight_layout()
plt.show()

# Second model with K=2
# define first centroid
centroid1 = random.choice(datas2.copy())
centroid2 = random.choice([x for x in datas2.copy() if x != centroid1])
# first plot
fig, it = plt.figure(), 0
ax = fig.add_subplot(projection='3d')
ax.scatter(dx, dy, dz, color='black', alpha=0.009)
ax.text2D(0.05, 0.95, 'Second model with K=2', transform=ax.transAxes)
ax.scatter(centroid1[0], centroid1[1], centroid1[2], color='red')
ax.scatter(centroid2[0], centroid2[1], centroid2[2], color='blue')
ax.set_xlabel('Room Type')
ax.set_ylabel('Price')
ax.set_zlabel('Minimum Nights')
plt.tight_layout()
plt.pause(1)
while True:
    clusredx, clusredy, clusredz = [], [], []
    clusblux, clusbluy, clusbluz = [], [], []
    cluster2 = []
    # clustering points
    for data in datas2.copy():
        cal1 = (data[0]-centroid1[0])**2 + (data[1] - centroid1[1])**2 + (data[2] - centroid1[2])**2
        cal2 = (data[0]-centroid2[0])**2 + (data[1] - centroid2[1])**2 + (data[2] - centroid2[2])**2
        if min(cal1, cal2) == cal1:
            clusredx.append(data[0])
            clusredy.append(data[1])
            clusredz.append(data[2])
            cluster2.append(0)
        elif min(cal1, cal2) == cal2:
            clusblux.append(data[0])
            clusbluy.append(data[1])
            clusbluz.append(data[2])
            cluster2.append(1)
    # define new centroid
    a, b = centroid1, centroid2
    centroid1 = [sum(clusredx)/len(clusredx), sum(clusredy)/len(clusredy), sum(clusredz)/len(clusredz)]
    centroid2 = [sum(clusblux)/len(clusblux), sum(clusbluy)/len(clusbluy), sum(clusbluz)/len(clusbluz)]
    # break condition
    if (a == centroid1) and (b == centroid2):
        #calculate SSE for clustering
        sse1, sse2 = 0, 0
        for i in range(len(clusredx)):
            sse1 = sse1 + ((clusredx[i]-centroid1[0])**2 + (clusredy[i] - centroid1[1])**2 + (clusredz[i] - centroid1[2])**2)
        for i in range(len(clusblux)):
            sse2 = sse2 + ((clusblux[i]-centroid2[0])**2 + (clusbluy[i] - centroid2[1])**2 + (clusbluz[i] - centroid2[2])**2)
        Sum2SSE = sse1 + sse2
        break
# define final plot
plt.cla()
ax.text2D(0.05, 0.95, 'Second model with K=2', transform=ax.transAxes)
ax.scatter(clusredx, clusredy, clusredz, color='red')
ax.scatter(clusblux, clusbluy, clusbluz, color='blue')
ax.set_xlabel('Room Type')
ax.set_ylabel('Price')
ax.set_zlabel('Minimum Nights')
plt.tight_layout()
plt.show()
# print SSE Score after clustering
print('Nilai SSE untuk model pertama : ',Sum1SSE)
print('Nilai SSE untuk model kedua : ',Sum2SSE)
print('')
# exporting dataset after clustering to csv and xlsx
dfs = list(zip(df['room_type'], df['price'], df['minimum_nights'], cluster1, cluster2))
dfs = pd.DataFrame(dfs, columns=['room_type', 'price', 'minimum_nights', 'Cluster1', 'Cluster2'])
dfs.to_csv (r'datasetExplore_csv.csv', index = False, header=True)
dfs.to_excel (r'datasetExplore_xlsx.xlsx', header=True)

# importing library for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Data preparation
X = df[['room_type', 'price', 'minimum_nights']]
y = [cluster1,cluster2]
# Classification
for i in range(2):
    xTrain, xTest, yTrain, yTest = train_test_split(X, y[i], test_size = 0.2, random_state=100)
    # using k-Nearest Neighbors for classification
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(xTrain, yTrain)
    y_pred = neigh.predict(xTest)
    print('NILAI AKURASI K-NEAREST NEIGHBORS')
    print('Nilai akurasi model ke-',i+1,' = ', accuracy_score(yTest, y_pred))
    print('Nilai F1 model ke-',i+1,' = ', f1_score(yTest, y_pred, average='macro'))
    print('Nilai Precision model ke-',i+1,' = ', precision_score(yTest, y_pred, average='macro'))
    print('Nilai Recall model ke-',i+1,' = ', recall_score(yTest, y_pred, average='macro'))
    print('')
    # using Naive Bayes for classification
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)
    y_pred = nb.predict(xTest)
    print('NILAI AKURASI NAIVE BAYES')
    print('Nilai akurasi model ke-',i+1,' = ', accuracy_score(yTest, y_pred))
    print('Nilai F1 model ke-',i+1,' = ', f1_score(yTest, y_pred, average='macro'))
    print('Nilai Precision model ke-',i+1,' = ', precision_score(yTest, y_pred, average='macro'))
    print('Nilai Recall model ke-',i+1,' = ', recall_score(yTest, y_pred, average='macro'))
    print('')