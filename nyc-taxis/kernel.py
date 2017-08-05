import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# taken from
# https://bigdatascientistblog.wordpress.com/2015/10/02/feature-engineering-with-dates-part-1/
month = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month
day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday()
day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day
hour = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour
minute = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute

seasons = [0, 0, 1, 1, 1, 2]
season = lambda x: seasons[
    (datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month - 1)]

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dr = pd.date_range(start='2016-01-01', end='2016-06-30')
df = pd.DataFrame()
df['Date'] = dr

cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

df['Holiday'] = df['Date'].isin(holidays)

# for train data
train['month'] = train['pickup_datetime'].map(month)
train['day'] = train['pickup_datetime'].map(day)
train['day_of_week'] = train['pickup_datetime'].map(day_of_week)
train['hour'] = train['pickup_datetime'].map(hour)
train['minute'] = train['pickup_datetime'].map(minute)
train['season'] = train['pickup_datetime'].map(season)

train['Date2'] = pd.to_datetime(train['pickup_datetime'], errors='coerce')
train['holiday'] = train['Date2'].dt.date.astype(
    'datetime64[ns]').isin(holidays)
train = train.drop(['Date2'], axis=1)

train['store_and_fwd_flag'] = train[
    'store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype(int)

# for test data
test['month'] = test['pickup_datetime'].map(month)
test['day_of_week'] = test['pickup_datetime'].map(day_of_week)
test['day'] = test['pickup_datetime'].map(day)
test['hour'] = test['pickup_datetime'].map(hour)
test['minute'] = test['pickup_datetime'].map(minute)
test['season'] = test['pickup_datetime'].map(season)

test['Date2'] = pd.to_datetime(test['pickup_datetime'], errors='coerce')
test['holiday'] = test['Date2'].dt.date.astype('datetime64[ns]').isin(holidays)

test['store_and_fwd_flag'] = test[
    'store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype(int)

test_id = test["id"]
test = test.drop(['id', 'pickup_datetime', 'Date2'], axis=1)
X = train.drop(['id', 'trip_duration', 'pickup_datetime',
                'dropoff_datetime'], axis=1)
Y = train['trip_duration']

RF = RandomForestRegressor(verbose=1, n_jobs=2)
RF.fit(X, Y)

scores = cross_val_score(RF, X, Y,
                         scoring="neg_mean_squared_error", cv=4,
                         verbose=1, n_jobs=-1)
rmse_scores = np.sqrt(-scores)
rmse_scores.mean()
rmse_scores.std()

Y_pred = RF.predict(test)
sub = pd.DataFrame()
sub['id'] = test_id
sub['trip_duration'] = Y_pred
sub.to_csv('RF.csv', index=False)
