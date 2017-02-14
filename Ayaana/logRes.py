import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

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

print("Shuffling data")
cv = cross_validation.ShuffleSplit(len(data), n_iter=50, test_size=0.1, train_size=0.9)

# print(len(data))
# print(len(classification))

# trainingX = np.matrix(data[:55000])
# trainingY = np.array(classification[:55000])

# testingX = np.matrix(data[55000:])
# testingY = np.array(classification[55000:])


print 'c = ' + str(0.01)
model = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', C=0.01, warm_start=True)
scores = cross_validation.cross_val_score(model, data, classification, cv=cv)
print(scores)
# model.fit(trainingX, trainingY)
# result = model.score(testingX, testingY)
# print(1-result)
