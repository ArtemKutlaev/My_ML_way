from sklearn.neighbors import KNeighborsClassifier
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors= 5, p= 2, metric= "minkowski")

knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

print(f'Точность: {accuracy}')