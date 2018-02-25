import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import stem
from nltk.stem.snowball import SnowballStemmer

np.set_printoptions(threshold=np.inf)



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
        #round negative polarity of the tweet to the closest multiple of 3
        print('NEGFeature'+str(tweet[0])+'-->'+str(int(3 * round(float(negPolarity)/3)))) #print work in progress
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
        #round positive polarity of the tweet to the closest multiple of 3
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


def buildClassificationVector(dfClassified):
    classificationColumn=[]
    
    for classifiedTweet in dfClassified.itertuples(index=False):
        print(classifiedTweet)
        if (classifiedTweet[3] == 'negative'):
            classificationColumn.append(-1)
        elif (classifiedTweet[3] == 'neutral'):
            classificationColumn.append(0)
        else:
            classificationColumn.append(1)    
    
    return (np.array(classificationColumn))


def testDataCreator(df, dfClassified):

# This function creates a new DataFrame with not classified examples 
    classifiedID = []
    dfTest = pd.DataFrame(columns = ['id','tweet_text','emoji'])

    for element in dfClassified.itertuples():

        classifiedID.append(element.id)
    
    for tweet in df.itertuples():
    
        if(tweet.id not in classifiedID):

            dfTest = dfTest.append({'id': tweet.id, 'tweet_text': tweet.tweet_text, 'emoji': tweet.emoji}, ignore_index = True)
    
    return dfTest

stemmer = SnowballStemmer("italian")

df = pd.read_csv('./cleanedData.tsv')
dfClassified = pd.read_csv('./classified.tsv')
dfTestSet = testDataCreator(df, dfClassified)


dfNeg = pd.read_csv('./lexicon/neg.words.txt',header=None,names=['word'])
dfPos = pd.read_csv('./lexicon/pos.words.txt',header=None,names=['word'])

gramVectorizer = CountVectorizer(min_df=5, ngram_range=(1, 1))
bowFitter = gramVectorizer.fit(df['tweet_text'].values.astype(str))

# build dataset features vectors
DataSBowTransformer = bowFitter.transform(df['tweet_text'].values.astype(str))
DataSFeaturesVecTmp = DataSBowTransformer.toarray()
#print(DataSFeaturesVecTmp)

# build training set features vectors
TrainSBowTransformer = bowFitter.transform(dfClassified['tweet_text'].values.astype(str))
TrainSFeaturesVecTmp = TrainSBowTransformer.toarray()

# build test set feature vectors 
TestSBowTransformer = bowFitter.transform(dfTestSet['tweet_text'].values.astype(str))
TestSFeaturesVecTmp = TestSBowTransformer.toarray()

#Build polarity for Data set 
DataSNegPolarityFeat=buildNegPolarityFeat(df, dfNeg)
DataSPosPolarityFeat=buildPosPolarityFeat(df, dfPos)
DataSEmojiPolarityFeat=buildEmojiPolarity(df)

#Build polarity for Test set
TestSNegPolarityFeat = buildNegPolarityFeat(dfTestSet, dfNeg)
TestsPosPolarityFeat = buildPosPolarityFeat(dfTestSet, dfPos)
TestSEmojiPolarityFeat = buildEmojiPolarity(dfTestSet)

#Build polarity for train set
TrainSNegPolarityFeat=buildNegPolarityFeat(dfClassified, dfNeg)
TrainSPosPolarityFeat=buildPosPolarityFeat(dfClassified, dfPos)
TrainSEmojiPolarityFeat=buildEmojiPolarity(dfClassified)
classificationFeat = buildClassificationVector(dfClassified)

#print(negPolarityFeat.shape, posPolarityFeat.shape, emojiPolarityFeat.shape)
DataSFeaturesVec=np.c_[DataSFeaturesVecTmp, DataSNegPolarityFeat, DataSPosPolarityFeat, DataSEmojiPolarityFeat]
TrainSFeaturesVec=np.c_[TrainSFeaturesVecTmp,TrainSNegPolarityFeat,TrainSPosPolarityFeat,TrainSEmojiPolarityFeat,classificationFeat]
TestSFeaturesVec = np.c_[TestSFeaturesVecTmp, TestSNegPolarityFeat, TestSPosPolarityFeat, TestSEmojiPolarityFeat]


# print(DataSFeaturesVec)
# print(TrainSFeaturesVec)
# print(TestSFeaturesVec)
