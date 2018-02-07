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

# ----TWEET'S TEXT PREPROCESSING ( 1st PHASE )----
#
# this script cleans data from sanremo-*-.tsv keeping just usefull attributes and properly formatting tweet's text:
#
# SYNOPSIS:
#
# captions, urls and hashtags are filtered out thanks to tweet preprocessor lib (http://preprocessor.readthedocs.org/).
# retweetted aren't considered beacuse of redundancy.
# text is put in lowercase.
# emoticons = emoticons + emoji : 
#   since has been shown (Read J.,2005) that "emoticons,which has the potential of being independent
#   of domain, topic and time" we exploited emoji's sentiment classification from 
#   P. Kralj Novak, J. Smailovic, B. Sluban, I. Mozetic and map "significant" semantically polarized emoticons
#   into three categories: __EMO_POS, __EMO_NEG and __EMO_NEUT.
# punctuation, stop words, > 2 repetitions, spaces, two-lenght words and numbers are removed.
# 
# with these filters the size of the text and therefore the space of the features for the 2nd phase is drastically reduced.

# known issues: progressbar stops at 99%



# set module p to handle (filter) from tweets : urls, hashtags and mentions
p.set_options(p.OPT.URL,p.OPT.HASHTAG,p.OPT.MENTION)

# fields that are retrived from original dataset
fields = ['tweet_id_str','tweet_text','tweet_retweeted']

# names of all the fields from original dataset 
fNames = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags',
           'tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name',
           'tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name',
           'tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description',
           'user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count',
           'user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']

# initialize mapping with the most used emoticons
emoticons = \
	[	
        ('__EMOT_POS',	[':-)', ':)', '(:', '(-:', ':-D', ':D', 'X-D', 'XD', 'xD', '<3',
                         ';-)', ';)', ';-D', ';D', '(;', '(-;', ':3', ':>', '8-)'
                        ,'8)',  ':-}', ':}', '=]', '=)', 'B^D', ':-))', '^_^', '^ω^'] )	,\
		('__EMOT_NEG',	[':-(', ':(', ':c', ':‑c', ':,(', ':"(', ':((', ':‑[',
                         ':[', ":'‑(", ":'(", 'D;', 'D=', 'Q_Q', 'Q.Q', '-.-', '-_-'] )	,\

        ('__EMOT_NEUT',  [':|', ':‑|', ':$', ':S', ":\ " ] ) ,\
    ]

# |records| from original dataset
nrows= 213823

# object stemmer for the stemmatization 
stemmer = SnowballStemmer("italian")


def process(df):
    # process records extracted from original dataset and store them in clenadedData.csv    
    
    # create a new DataFrame with text and emoji as fields 
    newDf = pd.DataFrame(columns=[fields[1],'emoji'])
    i = 0
    for row in df.itertuples():
        i+=1
        if row[3] == 'no':
            # if it's not a retweet process it and add to the new DataFrame
            cleanedText = clean(row[2])
            newDf.loc[row[1]] = [cleanedText, emojiFind(row[2])]
        
        printProgress(i)

    newDf.to_csv('./cleanedData.csv')



def printProgress(count):
    #print a progress bar on stdout (this way an user wouldnt suspect a loop) 
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(((count*50)/nrows)), int((count*100)/nrows)))
    sys.stdout.flush()
    


def emojiFind(txt):
    #returns the categories of emoticons that were found in the tweet
    emotiPresence = {'__EMOT_POS': False, '__EMOT_NEG': False, '__EMOT_NEUT' : False}

    #label emoji in the tweet
    for character in txt:
        try:
            character.encode("ascii")
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced == '':
                try:
                    if unicodedata.name(character) in emoticons[0][1]:
                        #check if emoji found belongs to  __EMOT_POS
                        emotiPresence['__EMOT_POS'] = True
                    elif unicodedata.name(character) in emoticons[1][1]:
                        #check if emoji found belongs to __EMOT_NEG
                        emotiPresence['__EMOT_NEG'] = True
                    elif unicodedata.name(character) in emoticons[2][1]:
                        #check if emoji found belongs to __EMOT_NEUT
                        emotiPresence['__EMOT_NEUT'] = True
                         
                except ValueError:
                   print("decoding error")
    
    #label emoticon in the tweet
    for word in txt:
        if word in emoticons[0][1]:
            #check if emoti found belongs to  __EMOT_POS
            emotiPresence['__EMOT_POS'] = True
        elif word in emoticons[1][1]:
            #check if emoti found belongs to __EMOT_NEG
            emotiPresence['__EMOT_NEG'] = True
        elif word in emoticons[2][1]:
            #check if emoti found belongs to __EMOT_NEUT
            emotiPresence['__EMOT_NEUT'] = True

    return [k for k,v in emotiPresence.items() if v == True]         



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



def clean(txt):
    # apply all previous defined filters
    return stemmatize(removeAddSpace(removeOneorTwo(removeStopWords(removeNumbers((replaceTwOrMore(removePunc(p.clean(demojify(txt))).lower())))))))


def stemmatize(txt):
    
    returnString = ""
    for word in txt.split():
        returnString += stemmer.stem(word) + " "
    return returnString

def classifyEmoji():
    # get emoji's sentiment classification from P. Kralj Novak, J. Smailovic, B. Sluban, I. Mozetic : Sentiment of Emojis
    # select only those which have a clear sentiment polarization 
    # map selected emoji to '__EMOT_POS', '__EMOT_NEG', '__EMOT_NEUT'
    page = requests.get('http://kt.ijs.si/data/Emoji_sentiment_ranking/')
    tree = html.fromstring(page.content)

    emojiName = tree.xpath("//table[@id='myTable']/tbody/tr//td[11]/text()")
    emojiPositiveRatio = [float(ratio) for ratio in tree.xpath("//table[@id='myTable']/tbody/tr//td[8]/text()")]
    emojiNegativeRatio = [float(ratio) for ratio in tree.xpath("//table[@id='myTable']/tbody/tr//td[6]/text()")]
    emojiNeutralRatio = [float(ratio) for ratio in tree.xpath("//table[@id='myTable']/tbody/tr//td[7]/text()")]
    emojiOccurrence = [int(ratio) for ratio in tree.xpath("//table[@id='myTable']/tbody/tr//td[4]/text()")]
    
    emojiData = list(zip(emojiName,emojiPositiveRatio,emojiNegativeRatio,emojiNeutralRatio,emojiOccurrence))
    
    for i in range(len(emojiData)):
         
         if emojiData[i][4] > 200:
            # EURISTICS: we only consider emoji which have been evaluated in the study a "consistent" (at least 200) number of times
            #            the polarization ratio must be at least 0.1 bigger than other classifications
            polarization = max(emojiData[i][1],emojiData[i][2],emojiData[i][3])
                              
            if polarization == emojiData[i][1]:
                #positive polarization
                if (abs(polarization - emojiData[i][2]) > 0.1) and (abs(polarization - emojiData[i][3]) > 0.1):
                    #map the name of the emoji (emojiData[i][0]) to __EMOT_POS
                    emoticons[0][1].append(emojiData[i][0])
           
            elif polarization == emojiData[i][2]:
                #negative polarization
                if (abs(polarization - emojiData[i][1]) > 0.1) and (abs(polarization - emojiData[i][3]) > 0.1):
                    # // // // // to __EMOT_NEG
                    emoticons[1][1].append(emojiData[i][0])
            else:
                #neutral polarization
                if (abs(polarization - emojiData[i][1]) > 0.1) and (abs(polarization - emojiData[i][2]) > 0.1):
                    # // // // // to __EMOT_NEUT
                    emoticons[2][1].append(emojiData[i][0])
                    
        


if __name__ == "__main__":

    with open('stopWords.json') as jsonData:
        stopWordSet = set(json.load(jsonData))

    classifyEmoji()

    df = pd.read_table('sanremo-2017-0.1.tsv',names=fNames,usecols=fields,header=None,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'})
    process(df)

