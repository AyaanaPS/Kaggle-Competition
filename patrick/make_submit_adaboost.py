
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Dtree
from sklearn.ensemble import AdaBoostClassifier as adaboost

## use the best method so far, produce submission files

datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'

print('loading data')
train = np.genfromtxt(datapath+'train2.csv',delimiter=',',dtype='str')
train = train[1:].astype('float')
train = np.random.permutation(train)

test08 = np.genfromtxt(datapath+'test_2008.csv',delimiter=',',dtype='str')[1:].astype('float')
test12 = np.genfromtxt(datapath+'test_2012.csv',delimiter=',',dtype='str')[1:].astype('float')
ids08 = test08[:,0].astype('int')
ids12 = test12[:,0].astype('int')

test2_08 = np.genfromtxt(datapath+'test2_08.csv',delimiter=',',dtype='str').astype('float')
test2_12 = np.genfromtxt(datapath+'test2_12.csv',delimiter=',',dtype='str').astype('float')

xtrain = train[:,:-1]
ytrain = train[:,-1]
trainsize = train.shape[0]

## !! not really validation set, used to train
validsize = int(0.3*trainsize)
xvalid = train[-validsize:,:-1]
yvalid = train[-validsize:,-1]

print('training adaboost')

clf = adaboost(
			base_estimator=Dtree(
								criterion = 'entropy',
								max_depth = 2,
								),
			n_estimators=85,
			learning_rate = 0.3,
			  )

clf = clf.fit(xtrain,ytrain)

trainerror = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / trainsize

print('training error: %.4f' % trainerror)

validerror = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / validsize
## ^^ not really, used validation data to train

print('~validation error: %.4f' % validerror)

ypred08 = clf.predict(test2_08).astype('int')
ypred12 = clf.predict(test2_12).astype('int')

submit08 = np.array([ list(ids08.astype('int')), list(ypred08) ]).T
submit12 = np.array([ list(ids12.astype('int')), list(ypred12) ]).T

np.savetxt(datapath+'submit08.csv',submit08,fmt='%1d',delimiter=',',header='id,PES1')
np.savetxt(datapath+'submit12.csv',submit12,fmt='%1d',delimiter=',',header='id,PES1')


