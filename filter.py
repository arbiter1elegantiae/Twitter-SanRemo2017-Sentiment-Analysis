from __future__ import print_function
import pandas as pd
import preprocessor as p
import numpy as np
import sys
import unicodedata
import string
import re
import json
from collections import OrderedDict
import requests
from unidecode import unidecode
from emoji.unicode_codes import UNICODE_EMOJI
from lxml import html
from nltk import stem
from nltk.stem.snowball import SnowballStemmer



fNames = ['tweet_id_str','tweet_text', 'singers']

fields = fNames

dfCleaned = pd.read_table('./cleanedData.tsv')
dfClassified = pd.read_table('./classified.tsv')

stemmer = SnowballStemmer("italian")

# stopWordSet = set(json.load('stopWords.json'))

def findParticipantsTweets(df):

    participantsDF = pd.DataFrame(columns=fNames)

    for tweet in df.itertuples():
        
        a,b,c,d,e,f = tweet.tweet_text.split(' ',6)
        newString = ''
        visited = []

        tweet_mentions = handleMentions(tweet.singers)
        tweet_hashtags = handleHashtag(tweet.singers)

        for at in tweet_mentions:
                
                tmpString = ''
                resultAt = str(mentionsToName(at))
    
                if (resultAt != ''):
                   tmpString = resultAt
                   newString =  tweet[3] + tmpString

        for hashtag in tweet_hashtags:

                resultHash = str(hashtagToName(hashtag))

                if (resultHash != ''):
                    newString = newString + tweet[3] + resultHash        

        for word in newString.split():
        
            if (word.lower() in nomiPartecipanti and word.lower() not in visited):
                visited.append(word.lower())
                tweet_text = clean(tweet.singers)
                participantsDF = participantsDF.append({'tweet_id_str': tweet.tweet_id_str,'tweet_text': tweet_text, 'singers': word.lower()}, ignore_index=True)
                

    participantsDF.to_csv('participants.tsv')
    print(participantsDF)

def demojify(txt):
    # remove emoji from tweet
    returnString = ""

    for character in txt:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                    returnString = returnString
                except ValueError:
                    returnString += '[x]'
    return returnString


def removePunc(txt):
    # remove punctuation
    return ''.join([word.strip(string.punctuation)+' ' for word in txt.split(" ")])



def replaceTwOrMore(txt):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", txt)



def removeAddSpace(txt):
    # Remove additional white spaces
    return re.sub('[\s]+', ' ', txt)



def removeStopWords(txt):
    # remove stop words taken from the json file
    return ''.join([word+' ' for word in txt.split() if word not in stopWordSet])



def removeNumbers(txt):
    # remove all the numbers
    resulTxt=''
    for word in txt:
        resulTxt+=''.join([i for i in word if not i.isdigit()])
    return resulTxt



def removeOneorTwo(txt):
    # remove every one-two characters word
    return ''.join([word+' ' for word in txt.split() if len(word)>2 ])


def stemmatize(txt):

    returnString = ""
    for word in txt.split():
        returnString += stemmer.stem(word) + " "
    return returnString


def clean(txt):
    # apply all previous defined filters
    return stemmatize(removeAddSpace(removeOneorTwo(removeStopWords(removeNumbers((replaceTwOrMore(removePunc(p.clean(demojify(txt))).lower())))))))

def handleHashtag(txt):
    # finds all hashtags in the tweet and removes #
    hashtags = re.findall(r"#(\w+)", txt)
    return hashtags

def handleMentions(txt):
   # finds all mentions in the tweet and removes @
   mentions = re.findall(r"@(\w+)", txt)
   return mentions 

def mentionsToName(txt):
    
    if(txt.lower() == 'frankgabbani'):
        return 'gabbani'
    elif (txt.lower() == 'lodocomello'):
        return 'comello'
    elif (txt.lower() == 'roncellamare'):
        return 'ron'
    elif (txt.lower() == 'fabriziomorooff'):
        return 'moro'
    elif (txt.lower() == 'metaermal'):
        return 'meta'
    elif (txt.lower() == 'fiorellamannoia'):
        return 'mannoia'
    elif (txt.lower() == 'michele_bravi'):
        return 'bravi'
    elif (txt.lower() == 'paolaturci'):
        return 'turci'
    elif (txt.lower() == 'sergiosylvestre'):
        return 'sylvestre'
    elif (txt.lower() == 'elodiedipa'):
        return 'elodie'
    elif (txt.lower() == 'biancaatzei'):
        return 'atzei'
    elif (txt.lower() == 'samuelofficial'):
        return 'samuel'
    elif (txt.lower() == 'michelezarrillo'):
        return 'zarrillo'
    elif (txt.lower() == 'marcomasini64'):
        return 'masini'
    elif (txt.lower() == 'chiara_galiazzo'):
        return 'chiara'
    elif (txt.lower() == 'alessiobernabei'):
        return 'bernabei'
    elif (txt.lower() == 'clementinoiena'):
        return 'clementino'
    elif (txt.lower() == '_gigidalessio_'):
        return "d'alessio"
    elif (txt.lower() == 'giusyferreri'):
        return 'ferreri'
    elif (txt.lower() == 'neslimusic'):
        return 'nesli'
    elif (txt.lower() == 'alice_paba'):
        return 'nesli'
    elif (txt.lower() == 'raigeofficial'):
        return 'raige'
    elif (txt.lower() == 'giulia_luzi'):
        return 'raige'

def hashtagToName(txt):

    if ('comello' in txt):
        return 'comello'
    elif ('gabbani' in txt):
        return 'gabbani'
    elif ('ron' in txt):
        return 'ron'
    elif ('moro' in txt):
        return 'moro'
    elif ('meta' in txt):
        return 'meta'
    elif ('mannoia' in txt):
        return 'mannoia'
    elif ('turci' in txt):
        return 'turci'
    elif ('bravi' in txt):
        return 'bravi'
    elif ('clementino' in txt):
        return 'clementino'
    elif ('sylvestre' in txt):
        return 'sylvestre'
    elif ('elodie' in txt):
        return 'elodie'
    elif ('atzei' in txt):
        return 'atzei'
    elif ('zarrillo' in txt):
        return 'zarrillo'
    elif ('chiara' in txt):
        return 'chiara'
    elif ('samuel' in txt):
        return 'samuel'
    elif ('masini' in txt):
        return 'masini'
    elif ('bernabei' in txt):
        return 'bernabei'
    elif ('gigi' in txt):
        return "d'alessio"
    elif ('ferreri' in txt):
        return 'ferreri'
    elif ('nesli' in txt):
        return 'nesli'
    elif ('paba' in txt):
        return 'nesli'
    elif ('raige' in txt):
        return 'raige'
    elif ('luzi' in txt):
        return 'raige'
    
    

if __name__ == "__main__":

        with open('partecipanti.json') as jsonDataPartecipanti:
            nomiPartecipanti = set(json.load(jsonDataPartecipanti))


        with open('stopWords.json') as jsonData:
            stopWordSet = set(json.load(jsonData))


        df = pd.read_table('./Dataset_and_details/sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'}, nrows = 250000)
        findParticipantsTweets(df)
