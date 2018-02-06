#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np
import os.path
import json
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

sys.path.append('../../code/python')
from RandomForest import RandomForest

def readFile(path):
	X = []
	Y = []

	f = open(path,'r')

	for row in f:
		entries = row.strip("\n").split(",")
		
		Y.append(int(entries[0]))
		x = [int(e) for e in entries[1:]]
		X.append(x)

	return np.array(X),np.array(Y)

# def readFile(XPath, YPath,N):
# 	X = []
# 	Y = []
# 	f = open(XPath, "rb")
# 	l = open(YPath, "rb")

# 	f.read(16)
# 	l.read(8)
# 	images = []

# 	for i in range(N):
# 		Y.append(ord(l.read(1)))
# 		x = []
# 		for j in range(28*28):
# 			x.append(ord(f.read(1)))
# 		X.append(X)
# 	return np.array(X),np.array(Y)

def main(argv):
	outPath = "./text"

	print("Reading train files")
	XTrain,YTrain = readFile("train.csv")
	#XTrain,YTrain = readFile("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	
	print("Reading test files")
	XTest,YTest = readFile("test.csv")
	
	#XTest,YTest = readFile("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)

	NTrees = [25]
	for ntree in NTrees:
		clf = RandomForestClassifier(n_estimators=ntree, n_jobs=4) 
		print("Fitting model on " + str(len(XTrain)) + " data points")
		clf.fit(XTrain,YTrain)
		
		print("Testing model on " + str(len(XTest)) + " data points")
		start = timeit.default_timer()
		YPredicted = clf.predict(XTest)
		end = timeit.default_timer()
		print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
		print("Accuracy:%s" % accuracy_score(YTest, YPredicted))
		print("Total time: " + str(end - start) + " ms")
		print("Throughput: " + str(len(XTest) / (float(end - start)*1000)) + " #elem/ms")

		print("Saving model to JSON on disk")
		forest = RandomForest.RandomForestClassifier(None)
		forest.fromSKLearn(clf,True)

		if not os.path.exists("text"):
			os.makedirs("text")

		with open("text/forest_"+str(ntree)+".json",'w') as outFile:
			outFile.write(forest.str())

		if not os.path.islink("text/forest_"+str(ntree)+"_test.csv"):
                        os.symlink("test.csv", "text/forest_"+str(ntree)+"_test.csv")

		print("Saving model to PKL on disk")
		joblib.dump(clf, "text/forest_"+str(ntree)+".pkl") 

		print("*** Summary ***")
		print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
		print(str(len(XTrain)+len(XTest)) + "\t" + str(len(XTrain[0])) + "\t" + str(accuracy_score(YTest, YPredicted)) + "\t" + str(forest.getAvgDepth()))
		
if __name__ == "__main__":
   main(sys.argv[1:])
