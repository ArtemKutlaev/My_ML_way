from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
sc = StandardScaler()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1, stratify= y)

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

