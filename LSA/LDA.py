import pandas as pd
from nlpia.data.loaders import get_data # Функция для получения наборов данных
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF векторизатор sklearn
from sklearn.preprocessing import MinMaxScaler # Нормализатор диапазона
from nltk.tokenize.casual import casual_tokenize # Токенайзер текстов разговорного типа
pd.options.display.width = 120 # Ширина вывода DataFrame (для красоты)

sms = get_data('sms-spam') # Берём набор данных
index = [f'sms{i}{"!" * j}' for (i, j) in zip(range(len(sms)), sms.spam)] # Создаём индексы для DataFrame и помечаем спамовые "!"
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['sms'] = sms.spam.astype(int) # Конвертируем bool значения в int

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize) # Берём TF-IDF векторизатор из sklearn
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray() # Обучаем (создаём веса) TF-IDF векторизатор

mask = sms.spam.astype(bool).values # Выбираем только спамовые строки
spam_centroid = tfidf_docs[mask].mean(axis=0) # Центрируем (берём среднее значение) спамовые веса векторизатора
ham_centroid = tfidf_docs[~mask].mean(axis=0) # Центрируем не спамовые веса (~ - логическа операция НЕ. Пример: ~[0, 1, 0, 0] == [1, 0, 1, 1])

spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid) # Проекции вектора между центроидами
spamminess_score.round(2) # Округление до двух знаков после точки

print(sms.text)

sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1, 1)) # Помещаем spamminess_score в диапазон 0-1
sms['lda_predict'] = (sms.lda_score > .5).astype(int) # Все очки, которые > .5 будут истинны, то есть равны 1
print(sms['spam lda_predict lda_score'.split()].round(2).head(6)) # Вывод результатов предсказания
print(f'{round(1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms), 3) * 100}% правильно классифицированных сообщений') # Эффективность