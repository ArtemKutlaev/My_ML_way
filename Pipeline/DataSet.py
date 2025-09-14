from sklearn import datasets
from sklearn.model_selection import train_test_split

load_breast = datasets.load_breast_cancer()

X = load_breast.data
y = load_breast.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, stratify=y, random_state=1)
