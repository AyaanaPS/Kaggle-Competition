## reformat the 2008 (or 2012) training and test data

## most fields have non-continuous labels, change these to N categorical vectors
## some fields have continuous-like labels (such as ascending age range) leave these as is

import numpy as np

datapath = '/Users/patrick-fischer/Documents/Caltech/machine_learning_2017/project1/'

print('loading data ...')

train = np.genfromtxt(datapath+'train_2008.csv',delimiter=',',dtype='str')
test08 = np.genfromtxt(datapath+'test_2008.csv',delimiter=',',dtype='str')
test12 = np.genfromtxt(datapath+'test_2012.csv',delimiter=',',dtype='str')

## separate string headers from integer data:

headers = train[0,:-1]	## column headers
## last is PES1 = target to predict. same for test
## first is id number

ytrain = train[1:,-1].astype('int')
train = train[1:,:-1].astype('int')	## convert from string, values are ids
test08 = test08[1:].astype('int')
test12 = test12[1:].astype('int')
## first column is id number
## first row is headers

##############################################

## note all columns with exactly two entries (leave these as is):
nEntries = np.array([ len(set(train[:,x])) for x in range(train.shape[1]) ])
twoEntries = np.where( nEntries == 2 )[0]

## all useful non-continuous-type columns (convert to categorical vectors):
categorical = np.array([
	4,	## final outcome code (?) e.g., completed, absent, to be demolished, ...
	6, 	## living quarters
	7,	## type of housing
	9,	## telephone elsewhere?
	10,	## telphone interview acceptable?
	18, ## household type
	20, ## type of interview
	24, ## own business or farm?
	29, ## region
	30, ## census state code (redundant with 31)
	31, ## (redundant with 30)
	34, 35,	## metropolitan / principal city
	36, ## must be paired with other
	39, ## relationship to reference person
	43, ## marital status
	45, ## sex, blank = removed by TA's, only 2 options
	46, ## served on active duty?
	47, ## still on armed forces?
	49, ## race
	50, ## detailed hispanic origin
	51, ## change in household composition
	52,53,54,55, ## ???
	56, ## hispanic?
	57, ## married / armed forces
	62, ## citizen?
	65, ## self reported or proxy (?)
	66, ## labor force recode
	67,68,69, ## work last week?
	74, ## retired?
	75, ## disabled?
	76, ## want a job?
	77,78, ## disability prevent work?
	79, ## have a job?
	80, ## on layoff?
	81, ## why absent from work?
	82, ## paid while absent?
	83, ## multiple jobs? (redundant with 84)
	87, ## usually work more than 35 hours?
	89, ## want to work fulltime?
	90, ## reason for working part time?
	91, ## reason not want to work full time?
	92, ## reason not work full time last week?
	93, ## lose any hours last week?
	95, ## extra hours last week?
	100, ## could have worked full time?
	109,110, ## given date to return to work?
	111,112, ## could have returned to work?
	113, ## looking for work?
	115, ## laid off from full or part time?
	119, ## how looking for work?
	138,139, ## could have started job?
	140,141, ## what were you doing before looking for work?
	142, ## when last worked
	144, ## looking for full or part time?
	145, ## want a job?
	146, ## reason not looking for work?
	147,148,149,150,151,152,158,159,160,161, ## other work questions
	164, ## reason not at work (98.6%, could cut out)
	165, 166, 168, 169, 170, 
	173, 174, ## redundant?
	176, 177, 178, 179, 183, 184, 185, 186, ## other work questions
	187, 188, 189, 194, 197, 198, 199, ## type of business/worker (195-196 are recodes)
	200, 201, 202, 203, ## detailed recodes, pretty long
	204, 205, 206,207,208, ## recodes
	209, ## major occupation category
	210, 211, ## recodes
	212, ## single or multiple jobs?
	213, ## ?
	214, ## overtime pay?
	215, 216, 217, ## pay-related questions
	230, 231, ## member of labor union or similar?
	232, ## when last worked?
	233, ## retired?
	234, ## situation?
	237, 238, 239, 240, ## enrolled in school?
	335, ## how got diploma?
	338, 339, ## education yes/no questions
	369, 370, 371, 372, 373, 374, 375, ## difficulty hearing, blind, other disabled
	])

## all columns with continuous-like entries (leave these as is):
continuous = np.array([
	11,  ## total family income, continuous from 1-16. -3,-2,-1 = refused, don't know, blank
	17,	 ## number of persons in household
	19,  ## (?) month in sample (?)
	21,	 ## number of attempted contacts
	37,	 ## city size, 0 = nonmetropolitan or not entered
	41,	 ## age, -1 = blank
	48,	 ## degree level, -1=blank, could consider one-hot vector
	64,  ## immigrant year of entry. 0 = u.s. born (-1 not included)
	84,  ## how many jobs? -1=blank
	85,	## hours worked per week. 0-99 continuous, -4=varies, -1=blank
	86,	## hours worked per week, other job. 0-99 continuous, -4=varies, -1=blank
	88, ## sum of hours and hours2
	94,	## hours taken off work. 0-99 continuous, -2=don't know, -1=blank
	96,	## hours added to work. 0-99 continuous, -2=don't know, -1=blank
	97,	## hours actually worked. 0-99 continuous, -3=refused, -2=don't know, -1=blank
	98,	## hours actually worked, other job. 0-99 continuous, -3=refused, -2=don't know, -1=blank
	99,	## hours actually worked, total. 0-99 continuous, -3=refused, -2=don't know, -1=blank
	114, ## duration of layoff, -1=blank,
	143, ## duration of job-seeking, -1=blank
	175, ## duration of unemployment
	215, ## periodicity of overtime pay, -1=blank
	219,220, ## ? rate of pay, -3,-2,-1 = refused, idk, blank
	223, ## usual hours, -1=blank, 0-99 continuous
	226,227, ## ? weekly overtime amount
	229, ## weeks paid per year, -1=blank
	241,242,243,244,245, ## ? weights ...
	246, ## children by age group, could be one-hot ...
	247, ## number of children younger than 18, -1=not a parent
	336,337,340, ## highest grade completed or similar, -1=not in universe
	347, ## ? final weight
	354,355,356,357, ## when did you serve in military?
	
	120, 167, 171	## special cases ... should format differently
	])

## make new training and test data:

train2 = []
test2_08 = []
test2_12 = []

## append columns, then transpose (could be better)

print('reformatting data ...')

for col in range(test08.shape[1]):
	
	if col in twoEntries:		## include as is
		train2.append(train[:,col])
		test2_08.append(test08[:,col])
		test2_12.append(test12[:,col])
		
	elif col in categorical:	## make each entry into a unique one-hot vector

		entries = set( list(set(train[:,col])) + list(set(test08[:,col])) + list(set(test12[:,col])) )

		for ee in entries:
			
			if np.sum( train[:,col] == ee ) < 10: ## ~ too few to justify added storage
				pass
			else:
				train2.append( (train[:,col] == ee).astype('int') )
				test2_08.append( (test08[:,col] == ee).astype('int') )
				test2_12.append( (test12[:,col] == ee).astype('int') )

	elif col in continuous:		## include as is
	#else:							## include as is
		train2.append(train[:,col])
		test2_08.append(test08[:,col])
		test2_12.append(test12[:,col])
		
		
##################
## special formats:

for col in [59,60,61]: ## country of birth, mother's country, father's country
	
	noanswer = (train[:,col] == -3)+(train[:,col]==-2)+(train[:,col]==-1)
	usborn = (train[:,col] == 57)
	usother = (train[:,col] == 66)+(train[:,col]==73)+(train[:,col]==78)+(train[:,col]==96)
	elsewhere = 1-(noanswer+usborn+usother)
	
	train2.append(noanswer.astype('int'))
	train2.append(usborn.astype('int'))
	train2.append(usother.astype('int'))
	train2.append(elsewhere.astype('int'))

	noanswer = (test08[:,col] == -3)+(test08[:,col]==-2)+(test08[:,col]==-1)
	usborn = (test08[:,col] == 57)
	usother = (test08[:,col] == 66)+(test08[:,col]==73)+(test08[:,col]==78)+(test08[:,col]==96)
	elsewhere = 1-(noanswer+usborn+usother)
	
	test2_08.append(noanswer.astype('int'))
	test2_08.append(usborn.astype('int'))
	test2_08.append(usother.astype('int'))
	test2_08.append(elsewhere.astype('int'))

	noanswer = (test12[:,col] == -3)+(test12[:,col]==-2)+(test12[:,col]==-1)
	usborn = (test12[:,col] == 57)
	usother = (test12[:,col] == 66)+(test12[:,col]==73)+(test12[:,col]==78)+(test12[:,col]==96)
	elsewhere = 1-(noanswer+usborn+usother)
	
	test2_12.append(noanswer.astype('int'))
	test2_12.append(usborn.astype('int'))
	test2_12.append(usother.astype('int'))
	test2_12.append(elsewhere.astype('int'))

test2_08 = np.array(test2_08).T
test2_12 = np.array(test2_12).T

train2.append(ytrain)
train2 = np.array(train2).T

np.savetxt(datapath+'test2_08.csv',test2_08, delimiter=',',fmt='%1d')
np.savetxt(datapath+'test2_12.csv',test2_12, delimiter=',',fmt='%1d')
np.savetxt(datapath+'train2.csv',train2, delimiter=',',fmt='%1d')

'''
############
## special cases -- mix of continuous and non (convert partially to one-hot vectors):

jobsearch = [120,121,122,123,124,125]	## ways searched for work
## ^^ combine (-2,-1,12,13) into one-hot vector, 1-11 into other one-hot vector

#js1 = np.sum( train[:,col] == 

howsearch = [126,127,128,129,130,131]	## how did you search for work?
## ^^ combine (-2,-1,10,11,12,13) into categorical vector, 1-9 into other one-hot vector
atwork = 167	## reason not at work or hours worked
## ^^ make -1-11 categorical vectors, 12-22 continuous
hoursatwork = 171	## make -1,7,8 separate categorical vectors, continuous 1-6
'''


