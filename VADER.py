from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sa = SentimentIntensityAnalyzer()
while True:
    print(sa.polarity_scores(text=input('Введите текст на аглийском: '))['compound'])