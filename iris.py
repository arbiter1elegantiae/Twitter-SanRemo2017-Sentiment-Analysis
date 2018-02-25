from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

print(iris.target)
# print(iris)

gnb = GaussianNB()

array1 = ([1, 2, 3], [4, 5, 6])
array2 = ([0, 0, 0], [1, 1, 1])
test = {'data': array1, 'target': array2}

print(test.data)
# y_pred = gnb.fit(test.data, test.target).predict(test.data)
# print("Number of mislabeled points out of a total %d points: %d"
#         % (test.data.shape[0]), (test.target != y_pred).sum())


# print(test)
# print('\n')
# print(iris)