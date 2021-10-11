import numpy as np
import pandas as pd

class Tokenizator:
    '''Simple tokenizer'''

    def __init__(self):
        self.vocab = []
        self.vocab_size = 0

    def tokenize(self, sentence: str, n_gramms: int=1):
        '''Tokenize sentence to onehot vectors'''

        token_sentence = sentence.split()

        n_grammed = []
        for i in range(len(token_sentence)):
            n_gramm = []
            for gramm in range(n_gramms):
                try:
                    n_gramm.append(token_sentence[i + gramm])
                except IndexError:
                    break
            
            n_grammed.append(' '.join(n_gramm))
        token_sentence = n_grammed

        num_tokens = len(token_sentence)

        self.vocab = sorted(set(token_sentence))
        self.vocab_size = len(self.vocab)

        onehot_vectors = np.zeros((num_tokens, self.vocab_size), int)
        for i, word in enumerate(token_sentence):
            onehot_vectors[i, self.vocab.index(word)] = 1

        return onehot_vectors

t = Tokenizator()
onehot_vector = t.tokenize('Привет, я Иннокентий. Как дела? Чем занимаешься?', 2)
df = pd.DataFrame(onehot_vector, columns=t.vocab)
print(df)