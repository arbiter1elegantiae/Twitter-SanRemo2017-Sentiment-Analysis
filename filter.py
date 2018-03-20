import pandas as pd
import numpy as np
import re
import json

import dataPreproc as dpp



newFields = ['id','tweet_text','emoji','singers']
columns = ['tweet_id_str', 'tweet_text', 'tweet_created_at']

dpp.classifyEmoji()

def handleMentions(txt):
   # finds all mentions in the tweet and removes @
   mentions = re.findall(r"@(\w+)", txt)
   return mentions



def handleHashtag(txt):
    # finds all hashtags in the tweet and removes #
    hashtags = re.findall(r"#(\w+)", txt)
    return hashtags



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
        return 'paba'
    elif (txt.lower() == 'raigeofficial'):
        return 'raige'
    elif (txt.lower() == 'giulia_luzi'):
        return 'luzi'
    else:
         return ''
    


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
        return 'paba'
    elif ('raige' in txt):
        return 'raige'
    elif ('luzi' in txt):
        return 'luzi'
    else:
        return ''

def findParticipantsTweets(df1):
   
   participantsDF = pd.DataFrame(columns=newFields)
   i = 0
   for tweet in df1.itertuples():
        
        if (int(tweet.tweet_id_str) < 830576620465430528): #last tweet before prizegiving
           
            tweet_mentions = handleMentions(tweet.tweet_text)
            tweet_hashtags = handleHashtag(tweet.tweet_text) 
            visited = []
            ats=''
            hashtags=''

            for at in tweet_mentions:
                ats += ' '+str(mentionsToName(at.lower())) if not '' else  ''
                
            for hashtag in tweet_hashtags:
                hashtags += str(hashtagToName(hashtag.lower())) if not '' else ''

            newString = hashtags+ ' ' + ats + ' '+tweet.tweet_text 

        for word in newString.split():
            
                if (word.lower() in nomiPartecipanti and word.lower() not in visited):
                    visited.append(word.lower())
                    tweet_text = dpp.clean(tweet.tweet_text)
                    participantsDF = participantsDF.append({'id': tweet.tweet_id_str,'tweet_text': tweet_text,'emoji' : dpp.emojiFind(tweet.tweet_text),'singers': word.lower()}, ignore_index=True)
                    
   participantsDF.to_csv('participants.tsv', index=False)
   
         

with open('partecipanti.json') as jsonDataPartecipanti:
            nomiPartecipanti = set(json.load(jsonDataPartecipanti))

df = pd.read_table('./sanremo-2017-0.1.tsv',header=None,usecols=columns,names=dpp.fNames)
df1 = df.iloc[1:] 

findParticipantsTweets(df1)