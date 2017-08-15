import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df_test.pickup_datetime = pd.to_datetime(df_test.pickup_datetime)
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(
    {'Y': 1, 'N': 0}).astype(int)
df_test['store_and_fwd_flag'] = df_test[
    'store_and_fwd_flag'].map({'Y': 1, 'N': 0}).astype(int)
df['pu_hour'] = df.pickup_datetime.dt.hour
df['weekday'] = df.pickup_datetime.dt.weekday
df_test['pu_hour'] = df_test.pickup_datetime.dt.hour
df_test['weekday'] = df_test.pickup_datetime.dt.weekday

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


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(124, input_dim=11,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(72,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(30,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(24,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(16,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(12,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


df['radial_distance'] = haversine_np(df.pickup_longitude, df.pickup_latitude,
                                     df.dropoff_longitude, df.dropoff_latitude)

df['direction'] = bearing_array(df.pickup_longitude, df.pickup_latitude,
                                df.dropoff_longitude, df.dropoff_latitude)

df_test['radial_distance'] = haversine_np(df_test.pickup_longitude,
                                          df_test.pickup_latitude,
                                          df_test.dropoff_longitude,
                                          df_test.dropoff_latitude)

df_test['direction'] = bearing_array(df_test.pickup_longitude,
                                     df_test.pickup_latitude,
                                     df_test.dropoff_longitude,
                                     df_test.dropoff_latitude)


# Removes the outliers

q99 = df.trip_duration.quantile(0.99)
df = df[df.trip_duration < q99]


df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)

y = df['log_trip_duration']
X = (
    df.passenger_count,
    df.pu_hour,
    df.radial_distance,
    df.weekday,
    df.pickup_latitude,
    df.pickup_longitude,
    df.dropoff_latitude,
    df.dropoff_longitude,
    df.store_and_fwd_flag,
    df.pickup_cluster,
    df.dropoff_cluster)
X = np.asarray(X)

y = y.values.reshape(y.shape[0], )
X = X.reshape(X.shape[1], X.shape[0])

X_test = (
    df_test.passenger_count,
    df_test.pu_hour,
    df_test.radial_distance,
    df_test.weekday,
    df_test.pickup_latitude,
    df_test.pickup_longitude,
    df_test.dropoff_latitude,
    df_test.dropoff_longitude,
    df_test.store_and_fwd_flag,
    df_test.pickup_cluster,
    df_test.dropoff_cluster)

X_test = np.asarray(X_test)
X_test = X_test.reshape(X_test.shape[1], X_test.shape[0])

seed = 1993
np.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model,
                                         epochs=30,
                                         batch_size=30,
                                         verbose=1,
                                         shuffle=True)))
pipeline = Pipeline(estimators)
pipeline.fit(X, y)

kfold = KFold(n_splits=4, random_state=1993)
results = cross_val_score(pipeline, X, y, cv=kfold)

s = pipeline.predict(X_test)

df_submit = df_test
df_submit['trip_duration'] = (np.exp(s) - 1).round()
df_submit[['id', 'trip_duration']].to_csv(
    'submission.csv.gz', index=False, compression='gzip')
