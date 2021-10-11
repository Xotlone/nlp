from collections import Counter

from nltk.tokenize import TreebankWordTokenizer

def term_frequency(text: str, word: str, tokenizer) -> float:
    tokens = tokenizer.tokenize(text.lower())

    bag_of_words = Counter(tokens)
    times = bag_of_words[word]
    num_unique_words = len(bag_of_words)
    tf = round(times / num_unique_words, 4)

    return tf

sentence = """The faster Harry got to the store, the faster Harry, the faster, would get home."""
tokenizer = TreebankWordTokenizer()

print(term_frequency(sentence, 'harry', tokenizer))