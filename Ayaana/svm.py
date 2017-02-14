from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
import csv
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn import cross_validation

classification = []
data = []

print("Loading data")
with open('train2.csv') as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		if(i != 0):
			classification.append(float(row[-1]))
			curData = row[1:-1]
			data.append([float(i) for i in curData])
		else:
			i+=1


print("Initializing and normalizing data")
# print(len(data))

random_state = np.random.RandomState(0)
print("Shuffling data")
cv = cross_validation.ShuffleSplit(len(data), n_iter=100, test_size=0.1, train_size=0.9)

print("Normalizing data")
# NORMALIZING: 
normalizedData = preprocessing.normalize(data, norm='l1')
data = preprocessing.scale(normalizedData)

# trainingX = data[:55000]
# normalizedTrainingX = preprocessing.normalize(trainingX, norm='l1')
# trainingX = preprocessing.scale(normalizedTrainingX)
# trainingY = classification[:55000]

# testingX = data[55000:]
# normalizedTestingX = preprocessing.normalize(testingX, norm='l1')
# testingX = preprocessing.scale(normalizedTestingX)
# testingY = classification[55000:]

# Method 1: 0.24 error
print("Building model")
# print("Method 1")
svc = OneVsRestClassifier(LinearSVC(random_state=0))
scores = cross_validation.cross_val_score(svc, data, classification, cv=cv)
print(scores)
# svc.fit(trainingX, trainingY)
# result = svc.score(testingX, testingY)
# print(1 - result)

# model = ExtraTreesClassifier()
# model.fit(trainingX, trainingY)
# print(model.feature_importances_)
