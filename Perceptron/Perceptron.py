import numpy as np
from DataSet import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score

class Perceptron(object):
    """
    Классификатор на основе персептрона

    Parameters:
        eta : float
            Скорость обучения между (0.0, 1.0)
        n_iter : int
            Количество прохода по обучающему набору данных(Количество эпох)
        random_state : int
            Начальное значение для генератора случайных чисел для инициализации весов начальными значениями
    Attributes:
        w_ : одномерный массив
            Веса после подгонки
        errors_ : список
            Количество неправильных классификаций(обновлений в каждой эпохе
    
    """
    
    def __init__(self, eta=0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Функция подгонки весов к обучающимся данным

        Args:
            X (подобие массива):  Обучающие значения, shape = [n_samples, n_features]
            y (подобие массива):   Целевые значения, shape = [n_samples]

        Returns:
            self: Объект self.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """
        Вычисляет общий вход
        Args:
            X (подобие массива): Один образец данных (массив признаков).

        Returns:
            float: Чистый входной сигнал.
        """   
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """
         Предсказывает метку класса.

         Args:
             X (подобие массива):  Один образец данных (массив признаков).

         Returns:
             int: Предсказанная метка класса (0 или 1).
         """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
ppn = Perceptron(eta= 0.1, n_iter= 20)

ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")
