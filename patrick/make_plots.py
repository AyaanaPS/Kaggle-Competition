
import numpy as np
import pickle
import matplotlib.pyplot as plt

savepath = '/Users/patrick-fischer/Dropbox/Caltech/machine_learning_2017/project1/'
datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'

####################################
## random forest:

docs = pickle.load(open(datapath+'train_results_forest.pkl','rb'))
docs = np.array(docs)

params = ['nEst','depth','maxf']
names = ['number of estimators','maximum decision tree depth','maximum number of features']

for i in range(len(params)):

	fig,ax = plt.subplots()

	ax.set_title('random forest errors vs.\n'+names[i])
	ax.set_xlabel(names[i])
	ax.set_ylabel('classification error \n(fraction of total)')
	ax.axhline( y = 0.25632, color='blue', label='predict all vote')
	
	ind = np.where([ doc['param']==params[i] for doc in docs ])[0]
	trainerr = np.array([ [ doc['value'] ] + list( doc['trainerrors'] ) for doc in docs[ind] ])
	validerr = np.array([ [ doc['value'] ] + list( doc['validerrors'] ) for doc in docs[ind] ])

	trainsort = np.argsort(trainerr[:,0])
	validsort = np.argsort(validerr[:,0])
	
	trainerr = trainerr[trainsort]
	validerr = validerr[validsort]
	
	ax.plot(trainerr[:,0],np.mean(trainerr[:,1:],axis=1),'-o',color='black',label='train error')
	ax.plot(validerr[:,0],np.mean(validerr[:,1:],axis=1),'-o',color='red',label='validation error')

	for kfold in range(5):
		ax.scatter( trainerr[:,0],trainerr[:,kfold+1],color='black',alpha=0.5 )	
		ax.scatter( validerr[:,0],validerr[:,kfold+1],color='red',alpha=0.5 )	

	ax.legend()

	#fig.tight_layout()
	fig.savefig(savepath+'plots_forest_'+params[i]+'.png')
	fig.show()

####################################
## adaboost:

docs = pickle.load(open(datapath+'train_results_adaboost.pkl','rb'))
docs = np.array(docs)

params = ['nEst','depth','rate']
optvals = [85,2,0.3]
names = ['number of estimators','maximum decision tree depth','learning rate']

for i in range(len(params)):

	fig,ax = plt.subplots()
	
	ax.set_title('AdaBoost errors vs.\n'+names[i])
	ax.set_xlabel(names[i])
	ax.set_ylabel('classification error \n(fraction of total)')
	ax.axhline( y = 0.25632, color='blue', label='predict all vote')
	ax.axvline( x = optvals[i], color='gray', label='optimal value' )
	
	ind = np.where([ doc['param']==params[i] for doc in docs ])[0]
	trainerr = np.array([ [ doc['value'] ] + list( doc['trainerrors'] ) for doc in docs[ind] ])
	validerr = np.array([ [ doc['value'] ] + list( doc['validerrors'] ) for doc in docs[ind] ])

	trainsort = np.argsort(trainerr[:,0])
	validsort = np.argsort(validerr[:,0])
	
	trainerr = trainerr[trainsort]
	validerr = validerr[validsort]
	
	ax.plot(trainerr[:,0],np.mean(trainerr[:,1:],axis=1),marker='.',color='black',label='train error')
	ax.plot(validerr[:,0],np.mean(validerr[:,1:],axis=1),marker='.',color='red',label='validation error')

	for kfold in range(5):
		ax.scatter( trainerr[:,0],trainerr[:,kfold+1],color='black',alpha=0.5 )	
		ax.scatter( validerr[:,0],validerr[:,kfold+1],color='red',alpha=0.5 )	

	ax.legend()

	#fig.tight_layout()
	fig.savefig(savepath+'plots_adaboost_'+params[i]+'.png')
	fig.show()



