import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

grams = {}
corpus =[] 
newDf = pd.DataFrame()

# def featureVectorBuilder():
# # select only those 1-gram that occurs more than 30 times in the dataset to bound feature space (also more domain specific)
#     df = pd.read_table('cleanedData.tsv')
#     for row in df.itertuples():
#         for key in row[1].split(',')[1].split(' '):
#           grams[key] = grams.get(key, 1) + 1       
          

# featureVectorBuilder()
# print([key for key in grams if grams[key] > 30])

def dfToList():
    vectorizer = CountVectorizer()
    df = pd.read_table('cleanedData.tsv', names = ['id'])
    for row in df.itertuples():
        grams = row[1].split(',')
        print(grams[1])
        corpus.append(grams[1])

    newDf['text'] = corpus  

    print(corpus)
    # print('\n\n\n\n\n\n\n\n')
    # print(newDf)

    X = vectorizer.fit_transform(corpus)
    print(X)


    print(vectorizer.get_feature_names())
    X.toarray()

    print(X)

if __name__ == "__main__":

    dfToList()

