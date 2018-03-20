import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
import features as feat
import _pickle as cPickle

# this script build and save the classifier
#
# SYNOPSIS:
#
# Multinomial Naive Bayesian classifier is the choice of the final model, taking
#   in account the annotated dataset.
# using grid search to tune parameters (in this case, weight classes) to find the best model.
# cross-validation is also used to validating the model.
# accuracy, precision, recall and f-measure are used to evaluate the model
#

# function that display the best results of a grid search
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# get the annotated dataset
classified = feat.getClassified()

# dividing labels and features
features = classified['data']
labels = classified['target']


classifier = MultinomialNB()

# grid search to find the best parameters (calculating with 3-fold cross-validation and measuring with f-measure)
param_grid = {"class_prior": [[0.3,0.33,0.4],[0.33,0.33,0.33], [0.4,0.33,0.3]], "fit_prior": [False]}
search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=2, scoring=make_scorer(accuracy_score))
search.fit(features,labels)
print(report(search.cv_results_, 3))

# set parameters finded that weighting more the negative tweets model get is more accurate
classifier = MultinomialNB(class_prior=[0.3, 0.33, 0.4], fit_prior=False)
# split dataset (33% test set, 67% training set)
xTrain, xTest, yTrain, yTest = train_test_split(features,labels,test_size=0.33)
# fit the model
model = classifier.fit(xTrain, yTrain)
preds = classifier.predict(xTest)

# Evaluate accuracy, precision, recall, and f-measure
print("accuracy:", accuracy_score(yTest, preds))
print("precision:", precision_score(yTest, preds, average='micro'))
print("recall:", recall_score(yTest, preds, average='micro'))
print("f-measure:", f1_score(yTest, preds, average='micro'))

# save the classifier
with open('./dumpedClassifier.pkl', 'wb') as fid:
    cPickle.dump(classifier, fid)
