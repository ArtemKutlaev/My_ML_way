from sklearn.ensemble import RandomForestClassifier
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score

forest= RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)

forest.fit(X_train_std, y_train)

y_pred = forest.predict(X_test_std)

accuracy = accuracy_score(y_test, y_pred)

print(f'Точность: {accuracy}')