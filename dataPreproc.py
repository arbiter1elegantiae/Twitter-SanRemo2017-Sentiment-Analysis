import pandas as pd
import preprocessor as p
import numpy as np
import sys
import unicodedata
import string
import re
import json
from unidecode import unidecode 
from emoji.unicode_codes import UNICODE_EMOJI



# ----FIRST FILTER PHASE----
#
# this script cleans data from sanremo-*-.tsv keeping just usefull attributes and properly formatting tweet's text:
# captions, urls and hashtags are filtered out thanks to tweet preprocessor lib (http://preprocessor.readthedocs.org/). also, retweetted aren't considered beacuse of redundancy and the text is put in lowercase.
# emoji are translated in words
# punctuation, stop words, > 2 repetitions, spaces, two-lenght words and numbers removed 
# pandas allows efficient handling for big (120mb) datasets
# 'fields' array contains the main features to carry on sentiment analysis 

#known issues: progressbar stops at 99%

p.set_options(p.OPT.URL,p.OPT.HASHTAG,p.OPT.MENTION)
fields = ['tweet_id_str','tweet_text','tweet_hashtags','tweet_retweeted']
f_names = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags','tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name','tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name','tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description','user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count','user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']
nrows= 213823


def process(df):
    
    newDf = pd.DataFrame(columns=fields[1:3]+['emoji'])
    i = 0

    for row in df.itertuples():
        i+=1
        if row[4] == 'no':
            cleanedText = clean(row[2])
            newDf.loc[row[1]] = [cleanedText, row[3],emojiFind(row[2])]
            
        printProgress(i)

    newDf.to_csv('./cleanedData.csv')

#print a progress bar on stdout (this way an user wouldnt suspect a loop) 
def printProgress(count):
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(((count*50)/nrows)), int((count*100)/nrows)))
    sys.stdout.flush()
    

#return the list of all emoji present inside the tweet
def emojiFind(txt):
    returnList = []

    for character in txt:
        try:
            character.encode("ascii")
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced == '':
                try:
                    returnList += [unicodedata.name(character)]
                except ValueError:
                    returnList += ['x']
    return returnList

#remove emoji from tweet
def demojify(txt):
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

#remove punctuation
def removePunc(txt):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', txt)


#look for 2 or more repetitions of character and replace with the character itself
def replaceTwOrMore(txt):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", txt)


#Remove additional white spaces
def removeAddSpace(txt):
    return re.sub('[\s]+', ' ', txt)


def removeStopWords(txt):
    return ''.join([word+' ' for word in txt.split() if word not in stopWordSet])

def removeNumbers(txt):
    resulTxt=''
    for word in txt:
        resulTxt+=''.join([i for i in word if not i.isdigit()])
    return resulTxt

#remove every one-two characters word
def removeOneorTwo(txt):
    return ''.join([word+' ' for word in txt.split() if len(word)>2 ])

def clean(txt):
    return removeAddSpace(removeOneorTwo(removeStopWords(removeNumbers((replaceTwOrMore(removePunc(p.clean(demojify(txt))).lower()))))))



with open('stopWords.json') as jsonData:
    stopWordSet = set(json.load(jsonData))

df = pd.read_table('sanremo-2017-0.1.tsv',names=f_names,usecols=fields,header=None,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object', fields[3]:'object'})
process(df)

