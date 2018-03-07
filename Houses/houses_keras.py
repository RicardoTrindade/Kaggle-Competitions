import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df.info()
df_test.info()

# Mapping no alley to 0, other values to +1 and -1
df['Alley'].value_counts()

df.drop(['Id'], axis=1)
df_test.drop(['Id'], axis=1)

# Probably dropping MiscFeature, Id,
# Converting all nulls to 0 on LotFrontage
df['LotFrontage'].value_counts()

pd.crosstab(df['YrSold'], df['MoSold'])
# May to June seem to be the best months

df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).astype(int)
df['CentralAir'].value_counts()

df['ExterQual'] = df['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}).astype(int)
df['ExterQual'].value_counts()

df['ExterCond'] = df['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}).astype(int)
df['ExterCond'].value_counts()

X = (df.LotArea,
     df.CentralAir,
     df.ExterCond,
     df.ExterQual,
     df.OverallQual,
     df.OverallCond,
     df.YearBuilt,
     df.YearRemodAdd,
     df.MoSold,
     df.YrSold)
X = np.asarray(X)
