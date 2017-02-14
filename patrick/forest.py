## python 3

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF

print('loading data')
datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'
train = np.genfromtxt(datapath+'train2.csv',delimiter=',',dtype='str')
train = train[1:].astype('float')
train = np.random.permutation(train)

trainfrac = 0.8
validfrac = 1.-trainfrac

trainsize = train.shape[0]*trainfrac
validsize = train.shape[0]*validfrac

if 'train_results_forest.pkl' in os.listdir(datapath):
	docs = pickle.load(open(datapath+'train_results_forest.pkl','rb'))
else: docs = []

#################################################

nEst = np.arange(5,300,20).astype('int')

print('training random forest vs number of estimators')

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
	
		clf = RF(
					criterion = 'entropy',
					max_depth = 50,
					max_features = 70,
					n_estimators=nEst[i],
					n_jobs=1
					  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	

	print('\t\t','validation error: %.4f' % np.mean(validerrors))	

	doc = { 'model': 'forest',
			'param': 'nEst',
			'value': nEst[i],
			'other': {'depth': 50, 'maxf': 70},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_forest.pkl','wb'))

#################################################

depths = np.arange(2.,100.,10.).astype('int')

print('training random forest vs decision tree depth')

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
	
		clf = RF(
					criterion = 'entropy',
					max_depth = depths[i],
					max_features = 70,
					n_estimators= 300,
					n_jobs=1
					  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	

	print('\t\t','validation error: %.4f' % np.mean(validerrors))	
	print('\t\t','training error: %.4f' % np.mean(trainerrors))	

	doc = { 'model': 'forest',
			'param': 'depth',
			'value': depths[i],
			'other': {'maxf': 70, 'nEst': 300},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_forest.pkl','wb'))

#################################################

maxf = np.arange(2,100,10).astype('int')

print('training random forest vs maximum features')

for i in range(len(maxf)):
	
	print(maxf[i])

	trainerrors = np.zeros(5)
	validerrors = np.zeros(5)

	for kfold in range(5):

		validind = list( range(round(validsize*kfold), round(validsize*(kfold+1))) )
		trainind = list(range(0,round(validsize*kfold))) + list(range(round(validsize*(kfold+1)),train.shape[0]))

		ytrain = train[trainind,-1]
		xtrain = train[trainind,:-1]

		yvalid = train[validind,-1]
		xvalid = train[validind,:-1]
	
		clf = RF(
					criterion = 'entropy',
					max_depth = 50,
					max_features = maxf[i],
					n_estimators= 300,
					n_jobs=1
					  )
	
		clf = clf.fit(xtrain,ytrain)
	
		trainerrors[kfold] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / float(trainsize)
		validerrors[kfold] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / float(validsize)
	
		print('\t','validation error: %.6f' % validerrors[kfold])	
		
	print('\t\t','validation error: %.4f' % np.mean(validerrors))	
	print('\t\t','training error: %.4f' % np.mean(trainerrors))	

	doc = { 'model': 'forest',
			'param': 'maxf',
			'value': maxf[i],
			'other': {'depth': 50, 'nEst': 300},
			'trainerrors': trainerrors,
			'validerrors': validerrors,
			}

	docs.append(doc)

	pickle.dump(docs,open(datapath+'train_results_forest.pkl','wb'))


