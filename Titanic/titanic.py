import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import RMSprop
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

df = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
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


batch_size = 128
num_classes = 2
epochs = 2

y = to_categorical(y, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(7,)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

ytest = model.predict(X_test)
df_test['Survived'] = ytest
print('Test shape OK') if df_test.shape[0] == ytest.shape[0] else print('Oops')
df_test[['PassengerId', 'Survived']].to_csv(
    'titanic_submission.csv.gz', index=False, compression='gzip')
