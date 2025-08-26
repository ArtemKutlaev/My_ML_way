from sklearn.svm import SVC
from Dataset import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score


svm = SVC(kernel='rbf', C=5.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

print(f'Точность: {accuracy}')


