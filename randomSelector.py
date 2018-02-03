import pandas as pd 
import numpy as np 

df = pd.read_csv('cleanedData.csv')
fields = ['tweet_id_str','tweet_text','tweet_hashtags','emoji']
randomDf = pd.DataFrame(columns = ['tweet_id_str','tweet_text','tweet_hashtags', 'classification'])

def addClassification():

    flag = True
    ans = '5'

    print('\n\n How would you classify the above tweet? ')
    print('''
        1)positive 
        2)neutral
        3)negative''')
    
    while flag == True and ans >= '4':
        ans = input('\n\nChoose your option: ' )
        if ans == '1':
            value = 1
            flag = False
        if ans == '2':
            value = 0
            flag = False
        if ans == '3':
            value = -1 
            flag = False 
    
    return value 


randomDf = df.sample(n=500)

for row in randomDf.itertuples():
    print('\n')
    print(row[2].upper())
    print(row[4])
    randomDf['Classification'] = addClassification()

print(randomDf)

randomDf.to_csv('classified.csv')
