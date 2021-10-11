# Латентное размещение Дирихле + Линейно-дискриминантный анализ для классификации спама из книги "Обработка естественного языка в действии"

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from nltk.tokenize import casual_tokenize
from nlpia.data.loaders import get_data

np.random.seed(42)

sms = get_data('sms-spam')
index = [f'sms{i}{"!" * j}' for  (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

counter = CountVectorizer(tokenizer=casual_tokenize) # Мультимножество
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index) # Мешок слов
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
bow_docs.columns = terms

ldia = LDiA(n_components=16, learning_method='batch') # Латентное размещение Дирихле с 16-ю темами
ldia = ldia.fit(bow_docs) # Обучение
columns = [f'topic{i}' for i in range(ldia.n_components)] # Номер колонки
components = pd.DataFrame(ldia.components_.T, index=terms, columns=columns) # Все значения тем
print(components.topic3.sort_values(ascending=False)[:10]) # Самые влиятельные слова/символы из четвёртой темы

ldia16_topic_vectors = ldia.transform(bow_docs) # Трансформация документов в вектора
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index, columns=columns) # Структурирование векторов
print(ldia16_topic_vectors.round(2).head())

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state=271828) # Создание датасета для линейно-дискриминантного анализа
lda = LDA(n_components=1) # Латентно-дискриминатнный анализатор с одной темой для спама
lda = lda.fit(X_train, y_train) # Обучение
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors) # Предсказание по векторам LDiA
print(round(float(lda.score(X_test, y_test)), 2)) # Точность

while True:
    query = input('Спам или не спам на английском: ')
    query = counter.transform(raw_documents=[query]).toarray() # Превращаем текст в мультимножество
    query = ldia.transform(query) # Трансформируем его в вектора LDiA
    predict = lda.predict(query)[0]
    out = 'спам' if predict else 'не спам'
    print(f'Вывод: {out}')