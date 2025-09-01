import pandas as pd  
from io import StringIO

CSV_data= \
'''A,B,C,D
1.0,2.0,3.0,4.О
5.0,6.0,,8.О
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(CSV_data))


# Удаляет все строки, где есть недостающее значение в столбце
df.dropna(axis=0)


# Условный расчет недостающих значений на основе среднего
df.fillna(df.mean())

# Реализацияф OneHotEncoder
pd.get_dummies(df[]) # в квадратных скобочках пишут название столбцов, которые хотят преобразовать