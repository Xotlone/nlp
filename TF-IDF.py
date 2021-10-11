from collections import Counter
from collections import OrderedDict
import copy
import math

from nltk.tokenize import TreebankWordTokenizer

def cosine_sim(vec1, vec2) -> float:
    '''Вычисление коэффициента Отиаи (коэффициент косинусного подобия)'''

    vec1 = list(vec1.values()) # Из словаря берём только значения частотности
    vec2 = list(vec2.values())

    dot_prod = 0
    for v1, v2 in zip(vec1, vec2):
        dot_prod += v1 * v2 # Скалярное умножение векторов

    mag_1 = math.sqrt(sum([x ** 2 for x in vec1])) # Нормализация вектора (квадратный корень суммы квадратов элементов). После нормализации вектор (список) превращается в число
    mag_2 = math.sqrt(sum([x ** 2 for x in vec2]))

    try:
        res = dot_prod / (mag_1 * mag_2) # Получаем косинус угла, то есть узнаём насколько наши векторы направлены в одну сторону
    except ZeroDivisionError:
        res = 0 # Угол 90 градусов
    return res

def tf_idf(t: Counter, d: int, D: list, zero_vector: OrderedDict) -> list:
    '''Вычисление TF-IDF'''

    vec = copy.copy(zero_vector)
    for key, val in t.items():
        docs_containing_key = 0
        for _doc in D:
            if key in _doc:
                docs_containing_key += 1
        tf = val / d # TF Частотность всех термов (слов)
        try:
            idf = len(D) / docs_containing_key # IDF Обратная частотность документов (слов в них)
        except ZeroDivisionError:
            idf = 0
        vec[key] = tf * idf # TF-IDF = TF * IDF Вычисление для каждого вектора в пространстве vec
    
    return vec

docs = [
    'Вещь может быть потребительной стоимостью и не быть стоимостью.',
    'Так бывает, когда ее полезность для человека не опосредствована трудом.',
    'Таковы: воздух, девственные земли, естественные луга, дикорастущий лес и т. д.'
] # Документы. Можно представить как все страницы на серверах поисковой системы

tokenizer = TreebankWordTokenizer()
doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))] # Токенизация каждого документа с понижением регистра
all_doc_tokens = sum(doc_tokens, [])
lexicon = sorted(set(all_doc_tokens)) # Все слова/символы, которые знает токенизатор

zero_vector = OrderedDict((token, 0) for token in lexicon) # Векторное пространство

doc_tfidf_vecs = [] # Документы но в формате TF-IDF (для будущего вычисления коэффициента Отиаи)
for doc in docs:
    tokens = tokenizer.tokenize(doc) # Токенизация каждого документа
    token_counts = Counter(tokens) # Подсчёт повтора слов

    doc_tfidf_vecs.append(tf_idf(token_counts, len(lexicon), docs, zero_vector)) # Вычисление TF-IDF

while True:
    query = input('Запрос: ')
    tokens = tokenizer.tokenize(query) # Токенизация запроса в цифры
    token_counts = Counter(tokens) # Подсчёт повторений слов

    query_vec = tf_idf(token_counts, len(tokens), docs, zero_vector) # Перевод в TF-IDF

    results = [cosine_sim(query_vec, i) for i in doc_tfidf_vecs] # Вычисление коэффициента Отиаи (подобия направлений векторов, схожести)
    output = ''
    for i in range(len(docs)):
        output += f'Подобие docs[{i}] вашему запросу = {results[i]} ({round(results[i] * 100, 1)}%)\n'
    print(output)
