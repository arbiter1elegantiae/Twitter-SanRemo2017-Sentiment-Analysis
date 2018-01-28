import pandas as pd
import preprocessor as p
#import numpy as np
import time
import ftfy

p.set_options(p.OPT.URL,p.OPT.HASHTAG,p.OPT.MENTION)
fields = ['tweet_id_str','tweet_text','tweet_hashtags','tweet_retweeted']

def process(df):
    for row in df.itertuples(index=False):
        if row[3] == 'no':
            tweet = p.clean(row[1])
            print(ftfy.fix_text(ftfy.fix_encoding(tweet)))
    
        
    


dataf = pd.read_csv('sanremo-2017-0.1.csv', usecols=fields, nrows=100)#, dtype={'tweet_id_str':np.uint64}
process(dataf)
