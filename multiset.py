import re
import pandas as pd

class Multiset:
    ''''Simple multiset of words'''

    def __init__(self):
        self.corpus = {} # Корпус двоичных векторов текстов
    
    def transform(self, sentences: str, filter: str=r'[-\s.,;!?]+'):
        '''Transform sentence to multiset'''

        for i, sent in enumerate(sentences.splitlines()): # Перечисление каждого предложения
            token_sent = re.split(filter, sent) # Фильтрация
            self.corpus[f'sent{i}'] = dict((tok, 1) for tok in token_sent) # Дополнение корпуса
        
        return pd.DataFrame.from_records(self.corpus).fillna(0).astype(int).T

sentences = '''Вещь может быть потребительной стоимостью и не быть стоимостью.
Так бывает, когда ее полезность для человека не опосредствована трудом.
Таковы: воздух, девственные земли, естественные луга, дикорастущий лес и т. д.'''
multiset = Multiset()
multiset_sentence = multiset.transform(sentences)
print(multiset_sentence)