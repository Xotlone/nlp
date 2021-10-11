import numpy as np

f = lambda z: np.exp(z) / np.sum(np.exp(z)) # Нормированная экспоненциальная функция для процентного рассчёта
print(f(np.array([.5, .9, .2])))