import pandas as pd

grams = {}

def featureVectorBuilder():
# select only those 1-gram that occurs more than 30 times in the dataset to bound feature space (also more domain specific)
    df = pd.read_table('cleanedData.tsv')
    for row in df.itertuples():
        for key in row[1].split(',')[1].split(' '):
          grams[key] = grams.get(key, 1) + 1       
          

featureVectorBuilder()
print([key for key in grams if grams[key] > 30])