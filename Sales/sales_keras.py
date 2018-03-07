import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

categories = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')

df = pd.read_csv('sales_train.csv.gz', compression='gzip')
# Pandas offers join methods like SQL