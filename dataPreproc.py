import pandas as pd
import preprocessor as p
import numpy as np
import sys
import unicodedata
from unidecode import unidecode 
from emoji.unicode_codes import UNICODE_EMOJI



# FIRST FILTER PHASE
# this script cleans data from sanremo-*-.tsv keeping just usefull attributes and properly formatting tweet's text:
# captions, urls and hashtags are filtered out thanks to tweet preprocessor lib (http://preprocessor.readthedocs.org/). also, retweetted aren't considered beacuse of redundancy and the text is put in lowercase. 
# pandas allows efficient handling for big (120mb) datasets
# 'fields' array contains the main features to carry on sentiment analysis 

#known issues: progressbar stops at 99%

p.set_options(p.OPT.URL,p.OPT.HASHTAG,p.OPT.MENTION)
fields = ['tweet_id_str','tweet_text','tweet_hashtags','tweet_retweeted']
f_names = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags','tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name','tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name','tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description','user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count','user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']
nrows= 213824

def process(df):
    
    newDf = pd.DataFrame(columns=fields[1:])
    i = 0

    for row in df.itertuples():
        i+=1
        if row[4] == 'no':
            test_string = demojify(row[2])
            #DA SISTEMARE, RIMPIAZZARE LE TEST_STRING CON IL TESTO DENTRO IL DATAFRAME
            newDf.replace(row[2], test_string)
            newDf.loc[row[1]] = [p.clean(row[2].lower()), row[3], row[4]]
            print('' + test_string)
        printProgress(i)

    print(newDf)    
   commented for testing
   #newDf.to_csv('./cleanedData.csv')

def printProgress(count):
    #print a progress bar on stdout (this way an user wouldnt suspect a loop) 
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(((count*50)/nrows)), int((count*100)/nrows)))
    sys.stdout.flush()
    

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
                    returnString += "[" + unicodedata.name(character) + "]"
                except ValueError:
                    returnString += "[x]"
    return returnString


df = pd.read_table('sanremo-2017-0.1.tsv',names=f_names,usecols=fields,header=None,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object', fields[3]:'object'})
process(df)

