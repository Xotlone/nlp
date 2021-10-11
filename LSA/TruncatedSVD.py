# Классификатор тем для спамовых и не спамовых СМС

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nlpia.data.loaders import get_data
from nltk.tokenize.casual import casual_tokenize

pd.options.display.max_columns = 12

sms = get_data('sms-spam') # Набор данных
index = [f'sms{i}{"!" * j}' for  (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean() # Центрирование векторизированных документов путём вычитания математического ожидания

pca = TruncatedSVD(n_components=16, n_iter=100) # 16 тем. 100 циклов обучения
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = [f'topic{i}' for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print('<{:=^64}>'.format('Векторы тем для sms'))
print(pca_topic_vectors)

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys()))) # Сортировка по количеству термов
print('<{:=^64}>'.format('Все термы'))
print(terms)

weights = pd.DataFrame(pca.components_, columns=terms, index=[f'topic{i}' for i in range(pca.n_components)])
print('<{:=^64}>'.format('Веса'))
print(weights)

print('<{:=^64}>'.format('Оценка классификации'))
pca_topic_vectors = (pca_topic_vectors.T / np.linalg.norm(pca_topic_vectors, axis=1)).T
print(pca_topic_vectors.iloc[:10].dot(pca_topic_vectors.iloc[:10].T).round(1))
print('''
   Изучение столбца sms0 (или строки sms0) показывает, что косинусные коэф-
фициенты сходства между sms0 и спамовыми сообщениями (sms2!, sms5!, sms8!, 
sms9!) — существенно меньше нуля. Вектор темы для sms0 заметно отличается от 
векторов тем спамовых сообщений.

    Проделывая то же самое для столбца sms2!, видим положительную корреляцию 
с другими спамовыми сообщениями. Семантика спамовых сообщений схожа, они 
посвящены одинаковым темам.
''')

while True:
    query = input('Запрос на английском: ').lower().split()
    for term in query:
        if term not in terms:
            query.remove(term)
    print(f'Обработанный запрос: {" ".join(query)}')
    
    topics = (weights[query] * 100).T.sum().round(1) # Список сумм весов для каждой темы. Чем больше число - тем больше предложение относится к теме
    print('Веса тем:')
    print(topics, '\n')
    print(f'Самая подходящая тема: topic{np.argmax(topics)}')