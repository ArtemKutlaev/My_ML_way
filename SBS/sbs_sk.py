from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from DataSet import X_train_std, X_test_std, y_train, y_test

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SequentialFeatureSelector(estimator=knn, n_features_to_select= 1, direction= 'backward', scoring= 'accuracy', cv=None)

sbs.fit(X_train_std, y_train)

indices = sbs.get_support(indices=True)

X_train_std = X_train_std[:, indices]
X_test_std = X_test_std[:, indices]

knn.fit(X_train_std, y_train)

y_pred = knn.predict(X_test_std)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность:{accuracy}')