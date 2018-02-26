import pandas as pd
import numpy as np



nrows= 213823

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
    'AlBano',
    'AlessioBernabei',
    'BiancaAtzei',
    'Chiara',
    'Clementino',
    'Elodie',
    'ErmalMeta',
    'FabrizioMoro',
    'FiorellaMannoia',
    'FrancescoGabbani',
    'Gigi',
    'GiusyFerreri',
    'LodovicaComello',
    'MarcoMasini',
    'MicheleBravi',
    'MicheleZarrillo',
    'Nesli', #e quella dopo
    #'AlicePaba',
    'PaolaTurci',
    'Raige', #e quella dopo
    #'GiuliaLuzi',
    'Ron',
    'Samuel',
    'SergioSylvestre',
]
# ha vinto gabbani co le scimmie

from features import *
"""
# ste cose le ho dovute mettere perche mi dava errori sui caratteri (utf)
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
"""
dfClassified = getClassified()
df = pd.read_table('./sanremo-2017-0.1.tsv',header=None,usecols=fields,names=fNames,dtype={fields[0]:'object', fields[1]:'object'}, error_bad_lines=False)

def preproc2():
    """
    creo la feature candidates con all inizio tutti 0 poi in base ai nome cambio il valore (da 1 a len dei nomi+1)
    """
    candidates = [0 for i in range(len(dfClassified))]
    for i in range(len(dfClassified['id'])):
        txt = str(df[df.tweet_id_str == str(dfClassified['id'][i])].tweet_text)
        for n in range(len(candidates_name)):
            if candidates_name[n] in txt: candidates[i] += (n+1)
