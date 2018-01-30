#!/usr/bin/env python3

import sys
import numpy as np

from sklearn.model_selection import KFold

from RandomForest import RandomForestClassifier as MyRandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

#import RandomForest

def main(argv):
	print("CLASSIFICATION TEST")
	data = np.genfromtxt("ionosphere.data", delimiter=",") 
	X = np.array(data[:,:-1])
	Y = np.array(data[:,-1])

	# RUN X-VAL
	xval = KFold(n_splits = 5)
	myScores = []
	skScores = []
	tmpScores = []
	cnt = 0
	for train,test in xval.split(X):
		print("\tX-Val run", cnt)
		myForest = MyRandomForestClassifier(10)
		skForest = RandomForestClassifier(n_estimators=10,bootstrap=False)
		tmpForest = MyRandomForestClassifier(10)

		print("\tFit on", len(train), " examples")			
		myForest.fit(X[train],Y[train])
		skForest.fit(X[train],Y[train])
		tmpForest.fromSKLearn(skForest)

		myPred = myForest.predict(X[test])
		skPred = skForest.predict(X[test])
		tmpPred = tmpForest.predict(X[test])

		# Check if trees are actuall the same
		assert sum(skPred - tmpPred) == 0
		
		subTrees = myForest.getSubTrees(0.8, 20)
		print(subTrees)

		# classification error
		myScores.append(sum(abs(myPred - Y[test])) / len(Y[test]))
		skScores.append(sum(abs(skPred - Y[test])) / len(Y[test]))
		
		print("\t\tmy error:", myScores[-1])
		print("\t\tsk error:", skScores[-1])
		
		cnt += 1

	print("\tmy accuracy:", (1-sum(myScores)/len(myScores))*100.0, "%")		
	print("\tsk accuracy:", (1-sum(skScores)/len(skScores))*100.0, "%")		

	print("\nREGRESSION TEST")
	data = np.genfromtxt("regression-datasets-housing.csv", delimiter=",") 
	X = np.array(data[:,:-1])
	Y = np.array(data[:,-1])

	# RUN X-VAL
	xval = KFold(n_splits = 5)
	myScores = []
	skScores = []
	tmpScores = []
	cnt = 0
	for train,test in xval.split(X):
		print("\tX-Val run", cnt)
		myForest = MyRandomForestClassifier(10)
		skForest = RandomForestClassifier(n_estimators=10,bootstrap=False)
		tmpForest = MyRandomForestRegressor(10)

		print("\tFit on", len(train), " examples")			
		myForest.fit(X[train],Y[train])
		skForest.fit(X[train],Y[train])
		tmpForest.fromSKLearn(skForest)

		myPred = myForest.predict(X[test])
		skPred = skForest.predict(X[test])
		tmpPred = tmpForest.predict(X[test])

		# Check if trees are actuall the same
		assert sum(skPred - tmpPred) == 0
		
		subTrees = myForest.getSubTrees(0.8, 20)
		print(subTrees)

		myScores.append(sum(pow(myPred - Y[test],2)) / len(Y[test]))
		skScores.append(sum(pow(skPred - Y[test],2)) / len(Y[test]))
		print("\t\tmy error:", myScores[-1])
		print("\t\tsk error:", skScores[-1])
		
		cnt += 1

	print("\tmy error:", sum(myScores)/len(myScores)*100.0, "%")		
	print("\tsk error:", sum(skScores)/len(skScores)*100.0, "%")		


if __name__ == "__main__":
	main(sys.argv[1:]);