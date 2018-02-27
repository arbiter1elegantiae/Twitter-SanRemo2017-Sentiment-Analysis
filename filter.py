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


def findParticipantsTweets(df):
    
    participantsDF = pd.DataFrame(columns=fNames)

    for tweet in df.itertuples():

        for word in tweet[3].split():

            if (word.lower() in nomiPartecipanti):
                participantsDF = participantsDF.append({'tweet_id_str': tweet.tweet_id_str,'tweet_text': tweet.singers, 'singers': word}, ignore_index=True)
    
    participantsDF.to_csv('participants.tsv')    
    print(participantsDF)

    print(tweet.singers)

if __name__ == "__main__":

        with open('partecipanti.json') as jsonDataPartecipanti:
            nomiPartecipanti = set(json.load(jsonDataPartecipanti))
 
        df = pd.read_table('sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'}, nrows = 4000)
        findParticipantsTweets(df)


