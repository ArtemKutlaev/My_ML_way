from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score
from sklearn import tree


tree_model = DecisionTreeClassifier(criterion= 'gini',max_depth= 4, random_state= 1)
tree_model.fit(X_train_std, y_train)

y_pred = tree_model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy}')

tree.plot_tree(tree_model)
plt.show()
