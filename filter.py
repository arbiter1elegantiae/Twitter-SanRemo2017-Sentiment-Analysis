from __future__ import print_function
import pandas as pd
import preprocessor as p
import numpy as np
import sys
import unicodedata
import string
import re
import json
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

        for word in tweet[3].split():

            if (word.lower() in nomiPartecipanti):
                
                tweet_text = clean(tweet.singers)
                participantsDF = participantsDF.append({'tweet_id_str': tweet.tweet_id_str,'tweet_text': tweet_text, 'singers': word}, ignore_index=True)
    
    participantsDF.to_csv('participants.tsv')    
    print(participantsDF)

    print(tweet.singers)


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


if __name__ == "__main__":

        with open('partecipanti.json') as jsonDataPartecipanti:
            nomiPartecipanti = set(json.load(jsonDataPartecipanti))

        with open('stopWords.json') as jsonData:
            stopWordSet = set(json.load(jsonData))
            
        df = pd.read_table('sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'}, nrows = 4000)
        findParticipantsTweets(df)


