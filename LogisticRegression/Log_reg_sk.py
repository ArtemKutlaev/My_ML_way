from sklearn.linear_model import LogisticRegression
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score


lg = LogisticRegression(penalty='l2', C=1, solver='lbfgs', random_state=1)
lg.fit(X_train_std, y_train)
y_pred = lg.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность: {accuracy}")