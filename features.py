 import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import stem
from nltk.stem.snowball import SnowballStemmer

np.set_printoptions(threshold=np.inf)


# global objects

stemmer = SnowballStemmer("italian")

df = pd.read_csv('../cleanedData.tsv')
dfNeg = pd.read_csv('./lexicon/neg.words.txt',header=None,names=['word'])
dfPos = pd.read_csv('./lexicon/pos.words.txt',header=None,names=['word'])

    # bag of words (BOW) fitter
gramVectorizer = CountVectorizer(min_df=5, ngram_range=(1, 1))
bowFitter = gramVectorizer.fit(df['tweet_text'].values.astype(str))


# features builder utility functions

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
        
        if (classifiedTweet[3] == 'negative'):
            classificationColumn.append(-1)
        elif (classifiedTweet[3] == 'neutral'):
            classificationColumn.append(0)
        else:
            classificationColumn.append(1)    
    
    return (np.array(classificationColumn))



def unseenDataCreator(df, dfClassified):
    
# This function creates a new DataFrame with not classified examples 
    classifiedID = []
    dfTest = pd.DataFrame(columns = ['id','tweet_text','emoji'])
    
    for element in dfClassified.itertuples():
    
        classifiedID.append(element.id)
    
    for tweet in df.itertuples():
        
        if(tweet.id not in classifiedID):
    
            dfTest = dfTest.append({'id': tweet.id, 'tweet_text': tweet.tweet_text, 'emoji': tweet.emoji}, ignore_index = True)
    
    return dfTest




# features extraction for both Classified and Unseen examples

def getClassified():
    
    classified = {}
    dfClassified = pd.read_csv('./classified.tsv')

    # build bow feature for classified set
    ClassifiedSBowTransformer = bowFitter.transform(dfClassified['tweet_text'].values.astype(str))
    ClassifiedSBowFeat = ClassifiedSBowTransformer.toarray()

    # build polarity features for classified set
    ClassifiedSNegPolarityFeat = buildNegPolarityFeat(dfClassified, dfNeg)
    ClassifiedSPosPolarityFeat = buildPosPolarityFeat(dfClassified, dfPos)
    ClassifiedSEmojiPolarityFeat = buildEmojiPolarity(dfClassified)

    # resulting features matrix
    ClassifiedSFeaturesVec=np.c_[ClassifiedSBowFeat,ClassifiedSNegPolarityFeat,ClassifiedSPosPolarityFeat,ClassifiedSEmojiPolarityFeat]

    # target vector
    classificationFeat = buildClassificationVector(dfClassified)

    classified['id'] = dfClassified.id
    classified['data'] = ClassifiedSFeaturesVec
    classified['target'] = classificationFeat

    return classified



def getUnseen():
    
    unseen = {}

    dfTmp = pd.read_csv('../cleanedData.tsv',nrows=200) #testing purpose, must del this line

    dfClassified = pd.read_csv('./classified.tsv')
    dfUnseen = unseenDataCreator(dfTmp, dfClassified)

    # build bow feature for unseen set 
    UnseenSBowTransformer = bowFitter.transform(dfUnseen['tweet_text'].values.astype(str))
    UnseenSBowFeat = UnseenSBowTransformer.toarray()
   
    # build polarity features for unseen set
    UnseenSNegPolarityFeat = buildNegPolarityFeat(dfUnseen, dfNeg)
    UnseenSPosPolarityFeat = buildPosPolarityFeat(dfUnseen, dfPos)
    UnseenSEmojiPolarityFeat = buildEmojiPolarity(dfUnseen)

    # resulting features matrix
    UnseenSFeaturesVec = np.c_[UnseenSBowFeat, UnseenSNegPolarityFeat, UnseenSPosPolarityFeat, UnseenSEmojiPolarityFeat]
    
    unseen['id'] = dfUnseen.id
    unseen['data'] = UnseenSFeaturesVec

    return unseen