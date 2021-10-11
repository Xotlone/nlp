# Реализация skip-грамм и непрерывного мультимножества слов CBoW (Continuous Bag of Words) на чистом Python (pandas нужен только для наглядности)
# Код реализован XOTLONE. Понятия skip-грамм и CBoW из книги "Обработка естественного языка в действии"

import pandas as pd

class SGaCBoW:
    '''Класс с лексисоном skip-грамм и CBoW для Word2Vec сети'''
    def __init__(self, lexicon: list):
        '''Инициализация лексиконов'''

        self.lexicon = lexicon
        self.dict_lexicon = {self.lexicon[i]: [int(j == i) for j in range(len(self.lexicon))] for i in range(len(self.lexicon))} # Лексикон с унитарными векторами
    
    def lexicon_add(self, words: list) -> None:
        '''Дополнение лексиконов'''

        for word in words:
            self.lexicon.append(word) if word not in self.lexicon else None
        self.dict_lexicon = {self.lexicon[i]: [int(j == i) for j in range(len(self.lexicon))] for i in range(len(self.lexicon))}

    def create(self, text: str, n_gramm: int=5) -> dict:
        '''Skip-грамма и CBoW из текста, слова которого присутствуют в лексиконе'''

        n_gramm -= 1 # Отнимаем единицу, так-как первой частью или последней будет просто слово
        tokenized_text = [i for i in text.split() if i in self.lexicon] # Токенизируем слова в текста, которые совпадают с лексиконом
        output = {'skip-gramm': [], 'CBoW': []} # Словарь с матрицами слов skip-грамм и непрерывного мультимножества
        for i, word in enumerate(tokenized_text): # Перечисляем все слова
            _tokenized_text = tokenized_text[:] # Копируем токенизированный текст, что-бы не менять исходный
            _tokenized_text.remove(word) # Убираем входное слово с текста
            
            gramm = [*list(range(-int(n_gramm / 2), 0)), *list(range(int(n_gramm / 2)))] # Создаём список индексов n-граммы ([-2, -1, 0, 1])
            frame = [] # Скользящая рамка, которая будет брать слова вокруг входного слова
            for g in gramm: # Перечисляем индексы n-граммы
                current_index = i + g # Индекс ближайшего слова к word = индекс word + индекс n-граммы
                if current_index >= 0 and current_index < len(_tokenized_text): # Если индекс ближайшего слова >= 0 и < длинны, то
                    frame.append(_tokenized_text[current_index]) # Добавляем это слова
                else: # Если нет, то
                    frame.append(None) # Добавляем None
            
            columns = { # Колонки для красоты. Здесь наглядно видно отличие skip-грамм от CBoW, и да они отличаются всего расположением последней и первой колонок.
                        # Вообще вся суть skip-грамм в том, что-бы предсказать окружающие слова по центральному, а CBoW - центральное по окружающим.
                'skip-gramm': [
                    'inp_word', # Первая колонка - word
                    *list(map(lambda x: 'out ' + str(x), gramm[:int(n_gramm / 2)])), # Половина колонки - половина индексов n-граммы
                    *list(map(lambda x: 'out ' + str(x + 1), gramm[int(n_gramm / 2):])) # Вторая половина колонки - вторая половина индексов n-граммы
                ],
                'CBoW': [
                    *list(map(lambda x: 'inp ' + str(x), gramm[:int(n_gramm / 2)])), # Половина колонки - половина индексов n-граммы
                    *list(map(lambda x: 'inp ' + str(x + 1), gramm[int(n_gramm / 2):])), # Вторая половина колонки - вторая половина индексов n-граммы
                    'out_word', # Последняя колонка - word
                ]
            }

            output['skip-gramm'].append([word, *frame]) # Первая колонка - сырое слово word, все последующие - слова вокруг word
            output['CBoW'].append([*frame, word]) # Тоже самое, но для CBoW наоборот

        onehot = {'skip-gramm': [], 'CBoW': []} # Словарь с матрицами унитарных векторов onehot
        for i in range(len(output['skip-gramm'])): # Перечисляем все индексы со словами
            onehot['skip-gramm'].append([])
            onehot['CBoW'].append([])
            for j in range(len(output['skip-gramm'][i])): # Перечисляем индекс со словами
                try:
                    onehot['skip-gramm'][-1].append(self.dict_lexicon[output['skip-gramm'][i][j]]) # Пробуем добавить в матрицу слово, перекодированное в унитарный вектор
                except KeyError: # Если слово = None
                    onehot['skip-gramm'][-1].append([0 for i in range(len(self.lexicon))]) # Добавляем пустой унитарный вектор

                try:
                    onehot['CBoW'][-1].append(self.dict_lexicon[output['CBoW'][i][j]]) # Всё тоже самое что и для skip-грамм
                except KeyError:
                    onehot['CBoW'][-1].append([0 for i in range(len(self.lexicon))])

        return {'output': onehot, 'view': output, 'columns': columns}

w2v = SGaCBoW('Claude Monet painted the Grand Canal of Venice in'.split())
w2v.lexicon_add(['1908']) # Для тестирования добавляем к словарю ещё одно слово
data = w2v.create('Claude Monet painted the Grand Canal of Venice in 1908') # Вызываем функцию для генерации
print(pd.DataFrame(data['view']['skip-gramm'], columns=data['columns']['skip-gramm']), '\n') # DataFrame со словами для наглядности
print(pd.DataFrame(data['output']['skip-gramm'], columns=data['columns']['skip-gramm']), '\n') # DataFrame с унитарными векторами для обучающей выборки
print(pd.DataFrame(data['view']['CBoW'], columns=data['columns']['CBoW']), '\n')
print(pd.DataFrame(data['output']['CBoW'], columns=data['columns']['CBoW']), '\n')