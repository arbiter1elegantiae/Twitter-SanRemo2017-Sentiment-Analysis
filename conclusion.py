import numpy as np
import pandas as pd
import scipy
import features as feat

# fields that are retrived from original dataset
fields = ['tweet_id_str','tweet_text','tweet_retweeted']

# names of all the fields from original dataset
fNames = ['tweet_id_str','tweet_created_at','tweet_text','tweet_source','tweet_hashtags',
           'tweet_urls','tweet_user_mentions','tweet_media','tweet_in_reply_to_screen_name',
           'tweet_in_reply_to_status_id','tweet_retweeted','tweet_retweeted_status_user_screen_name',
           'tweet_retweeted_status_id','user_id_str','user_name','user_screen_name','user_description',
           'user_url','user_followers_count','user_friends_count','user_created_at','user_statuses_count',
'user_profile_image_url','user_location','tweet_lang','tweet_favorite_count','tweet_retweet_count']

candidates_name = [
    "gabbani",
    "mannoia",
    "meta",
    "bravi",
    "turci",
    "sylvestre",
    "moro",
    "elodie",
    "atzei",
    "samuel",
    "zarrillo",
    "comello",
    "masini",
    "chiara",
    "bernabei",
    "clementino",
    "d'alessio",
    "ron",
    "ferreri",
    "nesli",
    "paba",
    "raige",
    "luzi"]


import sys
reload(sys)
sys.setdefaultencoding("utf-8")
classified = feat.getClassified()
features = classified['target']
participants = pd.read_csv('./participants.tsv')
newdf = []
dfpart = np.zeros((len(participants.tweet_id_str),2)).astype('str')
dfpart[:,0] = participants.tweet_id_str
dfpart[:,1] = participants.singers
for i in range(len(classified['id'])):
    if str(classified['id'][i]) in dfpart:
        singer = dfpart[dfpart[:,0] == str(classified['id'][i])][:,1][0]
        newdf.append([singer,classified['target'][i]])
newdf = np.array(newdf)
lenrank = len(candidates_name)
prevision = np.zeros((lenrank,)) # array con le probabilita per ogni cantante
for c in range(lenrank):
    for p in newdf[newdf[:,0] == candidates_name[c]][:,1]: # e' un "per ogni probabilita in base al cantante"
        prevision[c] = prevision[c] + float(p)*1/300 # calcolo la previsione equipesando i positivi e i negativi (i neutri non li considero)

predicted = np.argsort(-prevision) # ordino i risultati
print('official, predicted')
for i in range(lenrank):
    print(i+1, candidates_name[i], candidates_name[predicted[i]])
official = [i+1 for i in range(lenrank)]
# calcolo il coefficiente di spearman che usa anche nel paper
print(scipy.stats.spearmanr(official, predicted+1))
