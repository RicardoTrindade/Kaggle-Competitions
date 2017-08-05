import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import os

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
df_test['Embarked'].fillna('S', inplace=True)
df_test['Embarked'] = df_test['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2}).astype(int)
df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df_test['Has_Cabin'] = df_test["Cabin"].apply(
    lambda x: 0 if type(x) == float else 1)


encoder = LabelEncoder()
df_board = df['Embarked'].apply(str)
df_board_test = df_test['Embarked'].apply(str)
df_board_encoded = encoder.fit_transform(df_board)
df_board_test_encoded = encoder.fit_transform(df_board_test)
# df_test['encoded_embark'] = df_board_test_encoded
# df['encoded_embark'] = df_board_encoded
df['Age'].isnull().values.any()  # Checks for nans in age and returns true
df_test['Age'].isnull().values.any()  # Checks for nans in age and returns true
med = df['Age'].median()
med_test = df_test['Age'].median()

df['Age'].fillna(med, inplace=True)
df_test['Age'].fillna(med_test, inplace=True)

# Checks for nans in Fare and returns true
df_test['Fare'].isnull().values.any()
med = df_test['Fare'].median()
df_test['Fare'].fillna(med, inplace=True)

y = df['Survived']
X = df.drop(['Survived', 'PassengerId', 'Ticket',
             'Name', 'Cabin', 'Embarked'], axis=1)
X_test = df_test.drop(['PassengerId', 'Ticket',
                       'Name', 'Cabin', 'Embarked'], axis=1)


batch_size = 3
num_classes = 2
epochs = 50
y = to_categorical(y, num_classes)

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(7,)))
model.add(Dense(10, activation='relu', input_shape=(7,)))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
X = np.asarray(X)
y = np.asarray(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
history = model.fit(X, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

X_test = np.asarray(X_test)
X_test_scaled = scaler.fit_transform(X_test)
ytest = model.predict(X_test)
predict = np.round(model.predict(X_test))
titanic_sub=pd.concat([df_test[["PassengerId"]], predictions], axis = 1)
titanic_sub=titanic_sub.rename(columns={0:'Survived'})
titanic_sub.head()
titanic_sub[['PassengerId', 'Survived']].to_csv("titanic_sub.csv", index=False)