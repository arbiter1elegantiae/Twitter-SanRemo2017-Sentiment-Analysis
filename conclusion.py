import numpy as np
import pandas as pd
import scipy
import features as feat
import _pickle as cPickle

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


participants = pd.read_csv('./participants.tsv',nrows=200) # 200 testing purpose
trainPartiDf = participants.drop(['singers'], axis=1)
print(trainPartiDf)
featuresParticipants = feat.featureVectorize(trainPartiDf)

# load the classifier
with open('dumpedClassifier.pkl', 'rb') as fid:
    classifier = cPickle.load(fid)

preds = classifier.predict(featuresParticipants['data'])

classifiedParticipants = np.c_[participants['singers'],preds]

lenrank = len(candidates_name)
prevision = np.zeros((lenrank,))

# counting the positive and negative tweet for each singers and making ratio on every prevision
for c in range(lenrank):
    for p in classifiedParticipants[classifiedParticipants[:,0] == candidates_name[c]][:,1]:
        prevision[c] = prevision[c] + float(p)*1/len(classifiedParticipants)

# ordering the results
predicted = np.argsort(-prevision)

# comparing the ranking with the official one
print('official, predicted')
for i in range(lenrank):
    print(i+1, candidates_name[i], candidates_name[predicted[i]])
official = [i+1 for i in range(lenrank)]

#calculate spearman's coefficient to get correlation between the two rankings
print(scipy.stats.spearmanr(official, predicted+1))

