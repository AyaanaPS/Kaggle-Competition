from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
import csv

from sklearn import preprocessing
import numpy as np

print("Loading Data")
train = np.genfromtxt('train2.csv',delimiter=',',dtype='str')
test = np.genfromtxt('test2.csv',delimiter=',',dtype='str')

data = []
classification = []

print("Parsing data")
for row in train:
	curData = row[:-1]
	data.append([float(i) for i in curData])
	classification.append(float(row[-1]))

testData = []
print("Parsing test")
for row in test:
	testData.append([float(i) for i in row])

print("Normalizing data")
# #Normalizing and Scaling
normalizedData = preprocessing.normalize(data, norm='l1')
data = preprocessing.scale(normalizedData)

print("Normalizing test data")
normalizedTestData = preprocessing.normalize(testData, norm='l1')
testData = preprocessing.scale(normalizedTestData)

# seed = 7
# kfold = cross_validation.KFold(len(data), n_folds=10, random_state = seed)

#Creating the sub models
estimators = []

print("Building Logistic Regression Model")
model1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.01, warm_start=True)
estimators.append(('logistic', model1))

print("Building Random Forest Model")
model2 = RandomForestClassifier(n_estimators=100, max_features=20, warm_start=True)
estimators.append(('rf', model2))

print("Building SVM Model")
model3 = OneVsRestClassifier(LinearSVC(random_state=0))
estimators.append(('svm', model3))

print("Creating Voting Ensemble")
ensemble = VotingClassifier(estimators)
# results = cross_validation.cross_val_score(ensemble, data, classification, cv=kfold)
# print(results)
# print(results.mean)
ensemble.fit(data, classification)
print("Computing Prediction")
testResults = ensemble.predict(testData)

with open('result.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for prediction in testResults:
    	spamwriter.writerow(prediction)
