#Линейный дискриминантный анализ
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lda = LDA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state= 1, solver='lbfgs')
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std) 

lr.fit(X_train_lda, y_train)
y_pred = lr.predict(X_test_lda)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность:{accuracy}')
