## python 3

import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
'''
print('loading data')
datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'
train = np.genfromtxt(datapath+'train_2008.csv',delimiter=',',dtype='str')
train = train[1:].astype('float')
train = np.random.permutation(train)
'''
trainfrac = 0.7
trainsize = int(train.shape[0]*trainfrac)
validsize = train.shape[0] - trainsize

ytrain = train[:trainsize,-1]
xtrain = train[:trainsize,:-1]

yvalid = train[trainsize:,-1]
xvalid = train[trainsize:,:-1]

################################################

nodesizes = np.arange(35,161,10)	## best ~ 85
trainerrors = np.zeros(len(nodesizes))
validerrors = np.zeros(len(nodesizes))

print('training decision tree vs node size')

for i in range(len(nodesizes)):

	clf = tree.DecisionTreeClassifier(
						criterion = 'entropy',
						min_samples_leaf = int(nodesizes[i]),
						)
	clf = clf.fit(xtrain,ytrain)
	
	trainerrors[i] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / trainsize
	validerrors[i] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / validsize	

fig,ax=plt.subplots()
ax.set_title('decision tree')
ax.set_xlabel('node size')
ax.set_ylabel('classification error \n(fraction of total)')

ax.axhline( y = np.sum(yvalid==2)/len(yvalid) )	## if we guess that everyone voted
ax.plot( nodesizes, trainerrors, '-o', c='black', label='training set')
ax.plot( nodesizes, validerrors, '-o', c='red', label='validation set')

minind = np.where(validerrors==validerrors.min())[0][0]
ax.text( s='%.4f' % validerrors[minind], x=nodesizes[minind], y=validerrors[minind] )

ax.legend( framealpha = 0.3, numpoints=1, loc='upper left' )
fig.show()

#################################################
'''
depths = np.arange(2.,12.1)	## best = 6
trainerrors = np.zeros(len(depths))
validerrors = np.zeros(len(depths))

print('training decision tree vs depth')

for i in range(len(depths)):
	
	clf = tree.DecisionTreeClassifier(
						criterion = 'entropy',
						max_depth = int(depths[i]),
						)
	clf = clf.fit(xtrain,ytrain)
	
	trainerrors[i] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / trainsize
	validerrors[i] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / validsize	

fig,ax=plt.subplots()
ax.set_title('decision tree')
ax.set_xlabel('max depth')
ax.set_ylabel('classification error \n(fraction of total)')

ax.axhline( y = np.sum(yvalid==2)/len(yvalid), label='predict everyone votes')
ax.plot( depths, trainerrors, '-o', c='black', label='training set')
ax.plot( depths, validerrors, '-o', c='red', label='validation set')

minind = np.where(validerrors==validerrors.min())[0][0]
ax.text( s='%.4f' % validerrors[minind], x=depths[minind], y=validerrors[minind] )

ax.legend(framealpha=0.3,numpoints=1,loc='lower left')
fig.show()
'''
#################################################
'''
weights = [0.7,0.8,0.9,1.,1.1,1.2,1.3]
trainerrors = np.zeros(len(weights))
validerrors = np.zeros(len(weights))

print('training decision tree vs class weights')

for i in range(len(weights)):
	
	clf = tree.DecisionTreeClassifier(
							criterion = 'entropy',
							#max_depth = 6,
							min_samples_leaf = 85,
							class_weight = {1: 1, 2: weights[i]}
									  )
	clf = clf.fit(xtrain,ytrain)
	
	trainerrors[i] = np.sum( ~( ytrain == clf.predict(xtrain) ) ) / trainsize
	validerrors[i] = np.sum( ~( yvalid == clf.predict(xvalid) ) ) / validsize	

fig,ax=plt.subplots()
ax.set_title('decision tree')
ax.set_xlabel('class 2 weight / class 1 weight')
ax.set_ylabel('classification error \n(fraction of total)')

ax.axhline( y = np.sum(yvalid==2)/len(yvalid), label='predict everyone votes')
ax.plot( weights, trainerrors, '-o', c='black', label='training set')
ax.plot( weights, validerrors, '-o', c='red', label='validation set')

minind = np.where(validerrors==validerrors.min())[0][0]
ax.text( s='%.4f' % validerrors[minind], x=weights[minind], y=validerrors[minind] )

ax.legend(framealpha=0.3,numpoints=1,loc='lower left')
fig.show()
'''


