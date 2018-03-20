import features as feat
import _pickle as cPickle


# this module shows the performances of the retrieved classifier on unseen data
# note: getUnseen return a small set of the entire unseen set


unseen = feat.getUnseen()

# load the classifier
with open('dumpedClassifier.pkl', 'rb') as fid:
    classifier = cPickle.load(fid)

preds = classifier.predict(unseen['data'])

for i in range(len(unseen['id'])):
    print(unseen['id'][i], preds[i])