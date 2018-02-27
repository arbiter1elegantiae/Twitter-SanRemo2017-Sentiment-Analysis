import pandas as pd 
import numpy as np 
import random


#this module is used to classify (by hand) already preprocessed tweets.
#results are appended to classified.tsv


fields = ['tweet_id_str','tweet_text','tweet_retweeted']
fNames = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags',
           'tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name',
           'tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name',
           'tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description',
           'user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count',
           'user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']

dfClean = pd.read_csv('../cleanedData.tsv')
df = pd.read_table('../sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object', fields[2]:'object'})

dfClassified = pd.read_csv('./classified.tsv', index_col=0)
while True:

    randId = random.choice(dfClean.id)
    if randId not in dfClassified.index:

        tweeText = df.loc[df['tweet_id_str'] == str(randId)].to_records(index=False)[0][1]

        print ('\n'+tweeText)
        print('\nHow would you classify the tweet?')
        ans = input('''
            1)positive 
            2)neutral
            3)negative
            other)stop classifying''')
    
        if ans != '1' and ans != '2' and ans != '3':
            break

        condition = dfClean.loc[dfClean['id'] == randId].to_records(index=False)

        if ans == '1':
            dfClassified.loc[condition[0][0]] = [condition[0][1],condition[0][2],'positive']
        elif ans == '2':
            dfClassified.loc[condition[0][0]] = [condition[0][1],condition[0][2],'neutral']
        else:
            dfClassified.loc[condition[0][0]] = [condition[0][1],condition[0][2],'negative']    
                
                

dfClassified.to_csv('./classified.tsv')        
