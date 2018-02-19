import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('cleanedData.tsv')

gramVectorizer = CountVectorizer(min_df=5)
bowTransformer = gramVectorizer.fit_transform(df['tweet_text'].values.astype(str))


print(gramVectorizer.get_feature_names())