import pandas as pd 
import numpy as np 


from pandas import  DataFrame


df = pd.read_csv('./cleanedData.tsv')
dfClassified = pd.read_csv('./classified.tsv')
dfTest = pd.DataFrame(columns = ['id','tweet_text','emoji'])

difference = []
classifiedID = []

# for tweet in df.itertuples():
    
#     for element in dfClassified.itertuples():
        
#         if (tweet.id != element.id):
#             print(tweet.id)
#             difference = tweet.id
#             # dfTest = dfTest.append(rows, ignore_index=True)
            
# # print (difference)  
# print('\n\n\n\n\nend')


for element in dfClassified.itertuples():

    classifiedID.append(element.id)

print(classifiedID)

for tweet in df.itertuples():
    
    if(tweet.id not in classifiedID):

        print(tweet.id)
        dfTest = dfTest.append({'id': tweet.id, 'tweet_text': tweet.tweet_text, 'emoji': tweet.emoji}, ignore_index = True)

print(dfTest)

        
    