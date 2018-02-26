import numpy as np
import pandas as pd
import scipy
import features as feat

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

official = [9, 8, 6, 14, 17, 21, 7, 5, 2, 20, 15, 12, 13, 3, 1, 4]

"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
"""
classified = feat.getClassified()
features = classified['target']
# creo il dataset fittizio
randomFeature = np.random.randint(1,21, size=300)
newfeature = np.zeros((300,2))
newfeature[:,0] = features
newfeature[:,1] = randomFeature
#
prevision = np.zeros((21,)) # array con le probabilita per ogni cantante
for c in range(21):
    for p in newfeature[newfeature[:,1] == c][:,0]: # e' un "per ogni probabilita in base al cantante"
        prevision[c] = prevision[c] + p*1/300 # calcolo la previsione equipesando i positivi e i negativi (i neutri non li considero)

predicted = np.argsort(-prevision) # ordino i risultati
print 'official, predicted'
for i in range(len(official)):
    print i+1, candidates_name[official[i]], candidates_name[predicted[i]]

# calcolo il coefficiente di spearman che usa anche nel paper
print scipy.stats.spearmanr(official, predicted[:16])
