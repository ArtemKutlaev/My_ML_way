from sklearn.linear_model import Perceptron
from DataSet import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score

ppn = Perceptron(eta0= 0.1, random_state=1)

ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Точность: {accuracy}")