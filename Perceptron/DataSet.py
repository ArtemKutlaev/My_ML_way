from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

s =  r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' 
df = pd.read_csv(s,
                 header= None,
                 encoding= 'utf-8'
                 )


y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1) 
X = df.iloc[0:100, [0, 2]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1, stratify=y)




