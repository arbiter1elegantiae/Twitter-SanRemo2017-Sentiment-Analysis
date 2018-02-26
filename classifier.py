import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import features as feat

classified = feat.getClassified()

features = classified['data']
labels = classified['target']

xTrain, xTest, yTrain, yTest = train_test_split(features,labels,test_size=0.33,random_state=42)

classifier = MultinomialNB()
model = classifier.fit(xTrain, yTrain)
preds = classifier.predict(xTest)

# Evaluate accuracy
print(accuracy_score(yTest, preds))


# Test model on some unseen data

unseen = feat.getUnseen()

someUnseen = unseen['data'][:100]

preds = classifier.predict(someUnseen)

for i in range(100):
    
    print(unseen['id'][i], preds[i])
    