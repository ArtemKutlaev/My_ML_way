import numpy as np
from DataSet import X_train_std, X_test_std, y_train, y_test
from sklearn.metrics import accuracy_score


class LogisticRegressionGD (object):
    """Классификатор на основе логистической регрессии,
    использующий градиентный спуск.

    Параметры
    ----------
    eta : float
        Скорость обучения (между 0.0 и 1.0).

    n_iter : int
        Проходы по обучающему набору данных.
    random_state : int
        Начальное значение генератора случайных чисел
        для инициализации случайных весами.

    Атрибуты
    ---------
    w_ : одномерный массив
        Веса после подгонки.
    cost_ : list
        Значение логистической функции издержек в каждой эпохе.
    """
    def __init__(self, eta=0.1, n_iter=20, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Подгоняет к обучающим данным.

        Параметры
        ----------
        X : {подобен массиву}, форма = [n_examples, n_features]
            Обучающие векторы, где n_examples - количество образцов,
            n_features - количество признаков.
        y : подобен массиву, форма = [n_examples]
            Целевые значения.

        Возвращает
        -------
        self : object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc= 0.0, scale= 0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) -
                    ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Вычисляет общий вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Вычисляет логистическую сигмоидальную активацию"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Возвращает метку класса после единичного шага"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # эквивалентно:
        # return np.where(self.activation(self.net_input(X))
        
lg = LogisticRegressionGD()
lg.fit(X_train_std, y_train)
y_pred = lg.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")
