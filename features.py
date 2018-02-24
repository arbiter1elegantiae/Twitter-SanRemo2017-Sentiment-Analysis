import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import stem
from nltk.stem.snowball import SnowballStemmer




def buildNegPolarityFeat(df, dfNeg):
    
    negColumn=[]
    
    for tweet in df.itertuples(index=False):
        
        tweetList = str(tweet[1]).split(' ')
        del tweetList[-1] #last element is '', doesnt count then remove
        
        oldStem = ''
        negPolarity = 0
        for row in dfNeg.itertuples(index=False):
            stem = stemmer.stem(row[0])
            if oldStem != stem:

                if stem in tweetList:
                    negPolarity += 1
                oldStem = stem
        #round negative polarity of the tweet to the closest multiple of 5
        print('NEGFeature'+str(tweet[0])+'-->'+str(int(5 * round(float(negPolarity)/5)))) #print work in progress
        negColumn.append(int(3 * round(float(negPolarity)/3)))
    
    
    return np.array(negColumn)




def buildPosPolarityFeat(df, dfPos):
    
    posColumn=[]
    
    for tweet in df.itertuples(index=False):
        
        tweetList = str(tweet[1]).split(' ')
        del tweetList[-1] #last element is '', doesnt count then remove
        
        oldStem = ''
        posPolarity = 0
        for row in dfPos.itertuples(index=False):
            stem = stemmer.stem(row[0])
            if oldStem != stem:

                if stem in tweetList:
                    posPolarity += 1
                oldStem = stem
        #round positive polarity of the tweet to the closest multiple of 5
        print('POSFeature'+str(tweet[0])+'-->'+str(int(3 * round(float(posPolarity)/3)))) #print work in progress
        posColumn.append(int(3 * round(float(posPolarity)/3)))

    
    return np.array(posColumn)      


def buildEmojiPolarity(df):
    emojiColumn=[]

    for tweet in df.itertuples(index=False):
        
        emojiPolarity=0
        for emoji in tweet[2]:
            
            if emoji == '__EMOT_POS':
                emojiPolarity+=1
            elif emoji == '__EMOT_NEG':
                emojiPolarity-=1
        
        print('EMOFeature'+str(tweet[0])+'-->'+str(int(2 * round(float(emojiPolarity)/2)))) #print work in progress        
        emojiColumn.append(int(2 * round(float(emojiPolarity)/2)))
    
    return np.array(emojiColumn)


def buildClassificationVector(df, dfClassified):
    classificationColumn=[]

    for tweet in df.itertuples(index=False):

        for classifiedTweet in dfClassified.itertuples(index=False):

            if (tweet[0] == classifiedTweet[0]):
                if (classifiedTweet[3] == 'negative'):
                    classificationColumn.append((classifiedTweet[0], -1))
                if (classifiedTweet[3] == 'neutral'):
                    classificationColumn.append((classifiedTweet[0], 0))
                if (classifiedTweet[3] == 'positive'):
                    classificationColumn.append((classifiedTweet[0], 1))    

    return (np.array(classificationColumn))




stemmer = SnowballStemmer("italian")

df = pd.read_csv('./cleanedData.tsv')
dfClassified = pd.read_csv('./classified.tsv')

dfNeg = pd.read_csv('./lexicon/neg.words.txt',header=None,names=['word'])
dfPos = pd.read_csv('./lexicon/pos.words.txt',header=None,names=['word'])

gramVectorizer = CountVectorizer(min_df=5, ngram_range=(1, 1))
bowTransformer = gramVectorizer.fit_transform(df['tweet_text'].values.astype(str))

featuresVecTmp = bowTransformer.toarray()

negPolarityFeat=buildNegPolarityFeat(df, dfNeg)
posPolarityFeat=buildPosPolarityFeat(df, dfPos)
emojiPolarityFeat=buildEmojiPolarity(df)
classificationFeat = buildClassificationVector(df, dfClassified)

print(negPolarityFeat.shape, posPolarityFeat.shape, emojiPolarityFeat.shape)
print(classificationFeat.shape)
print(classificationFeat)
featuresVec=np.c_[featuresVecTmp, negPolarityFeat, posPolarityFeat, emojiPolarityFeat]


print(featuresVec)
