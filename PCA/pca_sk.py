from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score

pca = PCA(n_components= 2)
lr = LogisticRegression(multi_class='ovr', random_state= 1, solver='lbfgs')
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)
lr.fit(X_train_pca, y_train)
y_pred = lr.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy}')

