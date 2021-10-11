import time

from keras.layers import Dense, Conv1D, GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle

class SimpleTokenizer:
    def __init__(self, max_words=10000, forbidden_symbols='!@\"#№$;%^:&?*()_-+=[]{}\'\\|,.<>/', stop_words=[]):
        self.max_words = max_words
        self.forbidden_symbols = forbidden_symbols
        self.stop_words = stop_words
        self.lexicon = {}
    
    def preprocess(self, text):
        for forbidden_symbol in self.forbidden_symbols:
            text = text.lower().replace(forbidden_symbol, '')
        return text.split()

    def fit(self, texts):
        start_time = time.time()
        print('Tokenizer fitting start...')
        i = 0
        is_fitting = True
        for text in texts:
            if is_fitting == False:
                break
            text = self.preprocess(text)
            for word in text:
                if word not in list(self.lexicon.keys()) and word not in self.stop_words:
                    self.lexicon[word] = i
                    i += 1
                    if i == self.max_words - 1:
                        is_fitting = False
                    elif i % (self.max_words / 10) == 0:
                        print(f'Fitted {i / (self.max_words / 100)}%')
        print(f'Tokenizer fitted! {round(time.time() - start_time, 2)}s.')
    
    def decode(self, texts, pad=0):
        start_time = time.time()
        print('Tokenizer decoding start...')
        texts = [self.preprocess(text) for text in texts]
        if pad == 0:
            pad = len(texts[0])

        out = []
        for i, text in enumerate(texts):
            out.append([])
            for word in range(pad):
                if word < len(text):
                    if text[word] in list(self.lexicon.keys()):
                        out[-1].append(self.lexicon[text[word]] / len(self.lexicon))
                    else:
                        out[-1].append(0)
                else:
                    out[-1].append(0)
            if i % (len(texts) / 10) == 0:
                print(f'Decoded {i / (len(texts) / 100)}%')
        print(f'Tokenizer decoded texts! {round(time.time() - start_time, 2)}s.')
        return out

MAX_WORDS = 100
LEXICON_LEN = 10000

train_dataframe = pd.read_json('sarcasm1.json', lines=True)
valid_dataframe = pd.read_json('sarcasm2.json', lines=True)

tokenizer = SimpleTokenizer(LEXICON_LEN)
tokenizer.fit(train_dataframe.headline)

train_tokens = tokenizer.decode(train_dataframe.headline, MAX_WORDS)
with open('train_tokens.pickle', 'wb') as f:
    pickle.dump(train_tokens, f)