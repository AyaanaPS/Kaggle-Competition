from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import preprocessing

print("Loading data")
train = np.genfromtxt('train2.csv',delimiter=',',dtype='str')
test = np.genfromtxt('test2.csv',delimiter=',',dtype='str')

data = []
classification = []

print("Parsing data")
for row in train:
	curData = row[:-1]
	data.append([float(i) for i in curData])
	classification.append(float(row[-1]))

print("Normalizing data")
#Normalizing and Scaling
normalizedData = preprocessing.normalize(data, norm='l1')
data = preprocessing.scale(normalizedData)

print("Building data model")
seed = 7
num_trees = 100
max_features = 30
kfold = cross_validation.KFold(len(normalizedData), n_folds=10, random_state = seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
print("Finished initializing model")
results = cross_validation.cross_val_score(model, data, classification, cv=kfold)
print(results)
print(results.mean())
