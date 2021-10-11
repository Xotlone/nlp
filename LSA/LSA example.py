import numpy as np
import pandas as pd

docs = [
	'NYC is the Big Apple.',
	'NYC is known as the Big Apple.',
	'I love NYC!',
	'I wore a hat to the Big Apple party in NYC.',
	'Come to NYC. See the Big Apple!',
	'Manhattan is called the Big Apple.',
	'New York is a big city for a small cat.',
	'The lion, a big cat, is the king of the jungle.',
	'I love my pet cat.',
	'I love New York City (NYC).',
	'Your dog chased my cat.'
]

words = ['cat', 'dog', 'apple', 'lion', 'nyc', 'love']

themes = ['anti-animal', 'animal', 'dog-not lion', 'cats', 'apple', 'nyc'] # Примерные значения тем

tdm = {}
for word in words:
	tdm[word] = []
	for doc in docs:
		tdm[word].append(int(word in doc))
tdm = pd.DataFrame(list(tdm.values()), index=words, columns=range(len(docs))) # term-document matrix
print('<{:=^64}>'.format('Терм-документ'))
print(tdm)

U, s, Vt = np.linalg.svd(tdm)
# U - левые сингулярные векторы. Терм-тема
# s - сингулярные значения. Диагональная матрица о захваченной информации измерениями семантического векторного пространства
# Vt - правые сингулярные векторы. Документ-документ о частоте использований одинаковых тем в документах

print('\n<{:=^64}>'.format('Терм-тема'))
print(pd.DataFrame(U, index=tdm.index).round(2))

S = np.zeros((len(U), len(Vt)))
pd.np.fill_diagonal(S, s)
print('\n<{:=^64}>'.format('Сингулярные значения'))
print(pd.DataFrame(S).round(1))

print('\n<{:=^64}>'.format('Документ-Документ'))
print(pd.DataFrame(Vt).round(2))

def reconstruct_tdm(U, S, Vt, tdm): # Показывает, насколько возрастает погрешность при отбрасывании сингулярных векторов (тем)
	err = []
	for numdim in range(len(s), 0, -1):
		S[numdim - 1, numdim - 1] = 0 # Укарачиваем сингулярную матрицу
		reconstructed_tdm = U.dot(S).dot(Vt)
		err.append(np.sqrt(((reconstructed_tdm - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))) # Считаем ошибку с использованием нормализации
	
	return (reconstructed_tdm, err)
		
reconstructed_tdm, errors = reconstruct_tdm(U, S, Vt, tdm)
print('\n<{:=^64}>'.format('Ошибки при урезании тем'))
print(pd.DataFrame(np.array(errors).round(2), index=range(len(errors))), '\n')

print('Все темы: ', ', '.join(themes), '\n')
print('Все слова (слова подобраны вручную, а не взяты с лексикона токенизатора (которым в данном коде служит простой метод split) так-как этот код - лишь пример работы латентно-семантического анализа): ', ', '.join(words), '\n')

while True:
	query = input('Запрос на английском: ').lower()
	bow = np.array([int(i in query.split()) for i in words]) # Генерация массива терм-документ или по-простому мешка слов
	res = np.array(np.dot(bow, U.T)).round(2) # Скалярное умножение мешка слов на транспонированные левые сингулярные векторы
	print('Тема: ', themes[np.argmax(res)])