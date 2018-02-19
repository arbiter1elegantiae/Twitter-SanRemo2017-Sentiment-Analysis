import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('provaParole.tsv')

bigramVectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2))
bowTransformer = bigramVectorizer.fit_transform(df['tweet_text'].values.astype(str))

print(bigramVectorizer.get_feature_names())

print(bowTransformer.toarray())
