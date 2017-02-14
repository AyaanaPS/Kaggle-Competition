## python 3

import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Dtree
from sklearn.ensemble import AdaBoostClassifier as adaboost

print('loading data')
datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'
train = np.genfromtxt(datapath+'train2.csv',delimiter=',',dtype='str')
train = train[1:].astype('float')
train = np.random.permutation(train)

trainfrac = 0.8
validfrac = 1.-trainfrac

trainsize = train.shape[0]*trainfrac
validsize = train.shape[0]*validfrac

if 'train_results_adaboost.pkl' in os.listdir(datapath):
	docs = pickle.load(open(datapath+'train_results_adaboost.pkl','rb'))
else: docs = []

#################################################

nEst = np.arange(5,151,10).astype('int')

print('training adaboost vs number of estimators')

for i in range(len(nEst)):
	
	print(nEst[i])

	trainerrors = np.zeros(5)
	validerrors = np.zeros(5)

	for kfold in range(5):

		validind = list( range(round(validsize*kfold), round(validsize*(kfold+1))) )
		trainind = list(range(0,round(validsize*kfold))) + list(range(round(validsize*(kfold+1)),train.shape[0]))

		ytrain = train[trainind,-1]
		xtrain = train[trainind,:-1]

		yvalid = train[validind,-1]
		xvalid = train[validind,:-1]
	
		clf = adaboost(
				base_estimator=Dtree(
									criterion = 'entropy',
									max_depth = 2,
									),
				n_estimators= nEst[i],
				learning_rate = 0.3,
				  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	
		
	print('\t\t','validation error: %.4f' % np.mean(validerrors))	

	doc = { 'model': 'adaboost',
			'param': 'nEst',
			'value': nEst[i],
			'other': {'depth': 2, 'rate': 0.3},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_adaboost.pkl','wb'))

#################################################

depths = np.arange(1.,10.).astype('int')

print('training adaboost vs decision tree depth')

for i in range(len(depths)):
	
	print(depths[i])

	trainerrors = np.zeros(5)
	validerrors = np.zeros(5)

	for kfold in range(5):

		validind = list( range(round(validsize*kfold), round(validsize*(kfold+1))) )
		trainind = list(range(0,round(validsize*kfold))) + list(range(round(validsize*(kfold+1)),train.shape[0]))

		ytrain = train[trainind,-1]
		xtrain = train[trainind,:-1]

		yvalid = train[validind,-1]
		xvalid = train[validind,:-1]
	
		clf = adaboost(
				base_estimator=Dtree(
									criterion = 'entropy',
									max_depth = depths[i],
									),
				n_estimators = 85,
				learning_rate = 0.3,
				  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	
		
	print('\t\t','validation error: %.4f' % np.mean(validerrors))	

	doc = { 'model': 'adaboost',
			'param': 'depth',
			'value': depths[i],
			'other': {'nEst': 85, 'rate': 0.3},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_adaboost.pkl','wb'))

#################################################

rates = np.arange(0.5,1.01,0.1)

print('training adaboost vs learning rate')

for i in range(len(rates)):
	
	print(rates[i])

	trainerrors = np.zeros(5)
	validerrors = np.zeros(5)

	for kfold in range(5):

		validind = list( range(round(validsize*kfold), round(validsize*(kfold+1))) )
		trainind = list(range(0,round(validsize*kfold))) + list(range(round(validsize*(kfold+1)),train.shape[0]))

		ytrain = train[trainind,-1]
		xtrain = train[trainind,:-1]

		yvalid = train[validind,-1]
		xvalid = train[validind,:-1]
	
		clf = adaboost(
				base_estimator=Dtree(
									criterion = 'entropy',
									max_depth = 2,
									),
				n_estimators = 85,
				learning_rate = rates[i],
				  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	
		
	print('\t\t','validation error: %.4f' % np.mean(validerrors))	

	doc = { 'model': 'adaboost',
			'param': 'rate',
			'value': rates[i],
			'other': {'nEst': 85, 'depth': 2},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_adaboost.pkl','wb'))


