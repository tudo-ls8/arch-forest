#!/usr/bin/env python3

import sys
import numpy as np

from sklearn.model_selection import KFold

from RandomForest import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

#import RandomForest

def main(argv):
	print("\nREGRESSION TEST")
	data = np.genfromtxt("regression-datasets-housing.csv", delimiter=",") 
	X = np.array(data[:,:-1])
	Y = np.array(data[:,-1])
	Y = (Y-min(Y))/(max(Y)-min(Y))

	xval = KFold(n_splits = 5)
	myScores = []
	skScores = []
	tmpScores = []
	cnt = 0
	for train,test in xval.split(X):
		print("\tX-Val run", cnt)
		myForest = RandomForestRegressor(10, "SGD", min_splits=5)
		skForest = ExtraTreesRegressor(n_estimators=10,bootstrap=False)

		print("\tFit on", len(train), " examples")			
		myForest.fit(X[train],Y[train])
		skForest.fit(X[train],Y[train])
		print("FIT DONE")
		myPred = myForest.predict(X[test])
		skPred = skForest.predict(X[test])
		print("PRED DONE")

		myScores.append(sum(pow(myPred - Y[test],2)) / len(Y[test]))
		skScores.append(sum(pow(skPred - Y[test],2)) / len(Y[test]))
		print("\t\tmy error:", myScores[-1])
		print("\t\tsk error:", skScores[-1])
		
		cnt += 1

	print("\tmy error:", sum(myScores)/len(myScores)*100.0, "%")		
	print("\tsk error:", sum(skScores)/len(skScores)*100.0, "%")		


if __name__ == "__main__":
	main(sys.argv[1:]);