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



fNames = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags',
           'tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name',
           'tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name',
           'tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description',
           'user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count',
           'user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']

fields = fNames


def findParticipantsTweets(df):
    
    participantsDF = pd.DataFrame(columns=fNames)

    for tweet in df.itertuples():

        for word in tweet[3].split():

            if (word.lower() in nomiPartecipanti):
                participantsDF = participantsDF.append({'tweet_id_str': tweet.tweet_id_str,
                                                        'tweet_created_at': tweet.tweet_created_at,
                                                        'tweet_text': tweet.tweet_text, 
                                                        'tweet_source': tweet.tweet_source, 
                                                        'tweet_hashtags': tweet.tweet_hashtags, 
                                                        'tweet_urls': tweet.tweet_urls,
                                                        'tweet_user_mentions': tweet.tweet_user_mentions, 
                                                        'tweet_media': tweet.tweet_media, 
                                                        'tweet_in_reply_to_screen_name': tweet.tweet_in_reply_to_screen_name, 
                                                        'tweet_in_reply_to_status_id': tweet.tweet_in_reply_to_status_id,
                                                        'tweet_retweeted': tweet.tweet_retweeted,
                                                        'tweet_retweeted_status_user_screen_name': tweet.tweet_retweeted_status_user_screen_name,
                                                        'tweet_retweeted_status_id': tweet.tweet_retweeted_status_id,
                                                        'user_id_str': tweet.user_id_str,
                                                        'user_name': tweet.user_name,
                                                        'user_screen_name': tweet.user_screen_name,
                                                        'user_description': tweet.user_description,
                                                        'user_url': tweet.user_url,
                                                        'user_followers_count': tweet.user_followers_count,
                                                        'user_friends_count': tweet.user_friends_count,
                                                        'user_created_at': tweet.user_created_at,
                                                        'user_statuses_count': tweet.user_statuses_count,
                                                        'user_profile_image_url': tweet.user_profile_image_url,
                                                        'user_location': tweet.user_location,
                                                        'tweet_lang': tweet.tweet_lang,
                                                        'tweet_favorite_count': tweet.tweet_favorite_count,
                                                        'tweet_retweet_count': tweet.tweet_retweet_count}, ignore_index=True)
        
    print(participantsDF)


if __name__ == "__main__":

        with open('partecipanti.json') as jsonDataPartecipanti:
            nomiPartecipanti = set(json.load(jsonDataPartecipanti))
 
        df = pd.read_table('sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'}, nrows = 4000)
        findParticipantsTweets(df)


