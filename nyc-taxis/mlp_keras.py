import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df_test.pickup_datetime = pd.to_datetime(df_test.pickup_datetime)

df['pu_hour'] = df.pickup_datetime.dt.hour
df['weekday'] = df.pickup_datetime.dt.weekday
df_test['pu_hour'] = df_test.pickup_datetime.dt.hour
df_test['weekday'] = df_test.pickup_datetime.dt.weekday


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1)* np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(24, input_dim=8,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
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
q = df.trip_duration.quantile(0.99)
df = df[df.trip_duration < q]

df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)
plt.hist(df['trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()

y = df['trip_duration']
X = (
     df.passenger_count,
     df.pu_hour,
     df.radial_distance,
     df.weekday,
     df.pickup_latitude,
     df.pickup_longitude,
     df.dropoff_latitude,
     df.dropoff_longitude)
X = np.asarray(X)

corr_matrix = df.corr()
corr_matrix["radial_distance"].sort_values(ascending=False)
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
          df_test.dropoff_longitude)

X_test = np.asarray(X_test)
X_test = X_test.reshape(X_test.shape[1], X_test.shape[0])

seed = 42
np.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model,
                                         epochs=20,
                                         batch_size=100,
                                         verbose=1,
                                         shuffle=True)))
pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, y, cv=kfold)
pipeline.fit(X, y)

s = pipeline.predict(X_test).round()
df_submit = df_test
df_submit['trip_duration'] = s
df_submit[['id', 'trip_duration']].to_csv('sampel.csv', index=False)
