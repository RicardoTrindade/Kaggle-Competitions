import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df_all = pd.concat((pd.read_csv('train.csv'), pd.read_csv('test.csv')))
df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)
df_all['trip_duration_log'] = df_all['trip_duration'].apply(np.log)

X = np.vstack((df_all[['pickup_latitude', 'pickup_longitude']], 
               df_all[['dropoff_latitude', 'dropoff_longitude']]))

# Remove abnormal locations
min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]

pca = PCA().fit(X)
X_pca = pca.transform(X)

_, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

sample_ind = np.random.permutation(len(X))[:10000]

ax1.scatter(X[sample_ind,0], X[sample_ind,1], s=1, lw=0)
ax1.set_title('Original')

ax2.scatter(X_pca[sample_ind,0], X_pca[sample_ind,1], s=1, lw=0)
ax2.set_title('Rotated')