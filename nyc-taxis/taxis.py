import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MiniBatchKMeans


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df_test.pickup_datetime = pd.to_datetime(df_test.pickup_datetime)

df['pu_hour'] = df.pickup_datetime.dt.hour
df['weekday'] = df.pickup_datetime.dt.weekday
df_test['pu_hour'] = df_test.pickup_datetime.dt.hour
df_test['weekday'] = df_test.pickup_datetime.dt.weekday
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(
    {'Y': 1, 'N': 0}).astype(int)
df_test['store_and_fwd_flag'] = df_test[
    'store_and_fwd_flag'].map({'Y': 1, 'N': 0}).astype(int)
# df['trip_duration_log'] = df['trip_duration'].apply(np.log)
# Trip duration in log is normal

coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values,
                    df_test[['pickup_latitude', 'pickup_longitude']].values,
                    df_test[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(
    n_clusters=100, batch_size=10000).fit(coords[sample_ind])

df.loc[:, 'pickup_cluster'] = kmeans.predict(
    df[['pickup_latitude', 'pickup_longitude']])
df.loc[:, 'dropoff_cluster'] = kmeans.predict(
    df[['dropoff_latitude', 'dropoff_longitude']])
df_test.loc[:, 'pickup_cluster'] = kmeans.predict(
    df_test[['pickup_latitude', 'pickup_longitude']])
df_test.loc[:, 'dropoff_cluster'] = kmeans.predict(
    df_test[['dropoff_latitude', 'dropoff_longitude']])
####################################################

X = np.vstack((df[['pickup_latitude', 'pickup_longitude']],
               df[['dropoff_latitude', 'dropoff_longitude']]))

# Remove abnormal locations
min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
X = X[(X[:, 0] > min_lat) & (X[:, 0] < max_lat) &
      (X[:, 1] > min_lng) & (X[:, 1] < max_lng)]

pca = PCA().fit(X)
X_pca = pca.transform(X)

del X

df['pickup_pca0'] = pca.transform(
    df[['pickup_latitude', 'pickup_longitude']])[:, 0]
df['pickup_pca1'] = pca.transform(
    df[['pickup_latitude', 'pickup_longitude']])[:, 1]

df['dropoff_pca0'] = pca.transform(
    df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
df['dropoff_pca1'] = pca.transform(
    df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

##################################################################
X = np.vstack((df_test[['pickup_latitude', 'pickup_longitude']],
               df_test[['dropoff_latitude', 'dropoff_longitude']]))

# Remove abnormal locations
min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
X = X[(X[:, 0] > min_lat) & (X[:, 0] < max_lat) &
      (X[:, 1] > min_lng) & (X[:, 1] < max_lng)]

pca = PCA().fit(X)
X_pca = pca.transform(X)

df_test['pickup_pca0'] = pca.transform(
    df_test[['pickup_latitude', 'pickup_longitude']])[:, 0]
df_test['pickup_pca1'] = pca.transform(
    df_test[['pickup_latitude', 'pickup_longitude']])[:, 1]

df_test['dropoff_pca0'] = pca.transform(
    df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
df_test['dropoff_pca1'] = pca.transform(
    df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

del X
##################################################################


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
        np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


df['radial_distance'] = haversine_np(df.pickup_pca1,
                                     df.pickup_pca0,
                                     df.dropoff_pca1,
                                     df.dropoff_pca0)

# df['direction'] = bearing_array(df.pickup_longitude, df.pickup_latitude,
#                                 df.dropoff_longitude, df.dropoff_latitude)

df_test['radial_distance'] = haversine_np(df_test.pickup_pca1,
                                          df_test.pickup_pca0,
                                          df_test.dropoff_pca1,
                                          df_test.dropoff_pca1)

# df_test['direction'] = bearing_array(df_test.pickup_longitude,
#                                      df_test.pickup_latitude,
#                                      df_test.dropoff_longitude,
#                                      df_test.dropoff_latitude)


# Removes the outliers
q = df.trip_duration.quantile(0.99)
df = df[df.trip_duration < q]

y = df['trip_duration']
X = df.drop(['id', 'trip_duration', 'store_and_fwd_flag',
             'pickup_datetime', 'dropoff_datetime'], axis=1)

X_test = df_test.drop(['id', 'store_and_fwd_flag',
                       'pickup_datetime'], axis=1)

forrest_reg = GradientBoostingRegressor(warm_start=True, verbose=1,
                                    n_estimators=40,
                                    min_samples_leaf=5)

forrest_reg.fit(X, y)
# param_grid = [{'n_estimators': [3, 10, 30], 'max_depth': [1, 10, 20]}]
# grid = GradientBoostingRegressor(verbose=1)
# grid_search = GridSearchCV(grid, param_grid, cv=10)
# grid_search.fit(X, y)

# grid_search.best_estimator_
# grid_search.best_params_
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# feature_importances = grid_search.best_estimator_.feature_importances_
# feature_importances
# attributes = df.columns  # Not correct

s = forrest_reg.predict(X_test)
df_submit = df_test
df_submit['trip_duration'] = s
df_submit[['id', 'trip_duration']].to_csv('sampel.csv', index=False)
