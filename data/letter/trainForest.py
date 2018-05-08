#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np
import os.path
import json
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

sys.path.append('../../code/python')
from RandomForest import RandomForest

def readFile(path):
	f = open(path, 'r')
	X = []
	Y = []
	for row in f:
		entries = row.strip().split(",")
		x = [int(e) for e in entries[1:]]
		# Labels are capital letter has. 'A' starts in ASCII code with 65
		# We map it to '0' here, since SKLearn internally starts with mapping = 0 
		# and I have no idea how it produces correct outputs in the first place
		y = ord(entries[0]) - 65
		
		X.append(x)
		Y.append(y)

	return np.array(X), np.array(Y)

def trainModel(model, name, XTrain,YTrain, XTest, YTest):
	print("Fitting model on " + str(len(XTrain)) + " data points")
	model.fit(XTrain,YTrain)
	
	print("Testing model on " + str(len(XTest)) + " data points")
	start = timeit.default_timer()
	YPredicted = model.predict(XTest)
	end = timeit.default_timer()
	print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
	print("Accuracy:%s" % accuracy_score(YTest, YPredicted))
	print("Total time: " + str(end - start) + " ms")
	print("Throughput: " + str(len(XTest) / (float(end - start)*1000)) + " #elem/ms")

	# print("Saving model to JSON on disk")
	# forest = RandomForest.RandomForestClassifier(None)
	# forest.fromSKLearn(model,True)

	# if not os.path.exists("text"):
	# 	os.makedirs("text")

	# with open("text/"+name+".json",'w') as outFile:
	# 	outFile.write(forest.str())

	# print("Saving model to PKL on disk")
	# joblib.dump(model, "text/"+name+".pkl") 

	# print("*** Summary ***")
	# print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
	# print(str(len(XTrain)+len(XTest)) + "\t" + str(len(XTrain[0])) + "\t" + str(accuracy_score(YTest, YPredicted)) + "\t" + str(forest.getAvgDepth()))

def main(argv):
	outPath = "./text"
	
	X,Y = readFile("letter-recognition.data")

	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)

	with open("test.csv", 'w') as outFile:
		for x,y in zip(XTest, YTest):
			line = str(y)
			for xi in x:
				line += "," + str(xi)

			outFile.write(line + "\n")

	#NTrees = [1,5,10,25,50,100,200]
	NTrees = [1]
	for ntree in NTrees:

		# print("Random Forest depth 2")
		# model = RandomForestClassifier(n_estimators=ntree, n_jobs=4,max_depth=2) 
		# trainModel(model, "RandomForest_depth_2_"+str(ntree), XTrain, YTrain, XTest, YTest)
		
		# print("Random Forest depth 4")
		# model = RandomForestClassifier(n_estimators=ntree, n_jobs=4,max_depth=4) 
		# trainModel(model, "RandomForest_depth_4_"+str(ntree), XTrain, YTrain, XTest, YTest)

		# print("Random Forest depth 8")
		# model = RandomForestClassifier(n_estimators=ntree, n_jobs=4,max_depth=8) 
		# trainModel(model, "RandomForest_depth_8_"+str(ntree), XTrain, YTrain, XTest, YTest)
		
		# print("Random Forest depth unlimited")
		# model = RandomForestClassifier(n_estimators=ntree, n_jobs=4) 
		# trainModel(model, "RandomForest_depth_unlimited_"+str(ntree), XTrain, YTrain, XTest, YTest)

		# print("Extra Trees")
		# model = ExtraTreesClassifier(n_estimators=ntree, n_jobs=4)
		# trainModel(model, "ExtraTrees_"+str(ntree), XTrain, YTrain, XTest, YTest)
		
		# print("Boosting depth 1")
		# model = GradientBoostingClassifier(n_estimators=ntree, max_depth = 1)
		# trainModel(model, "GBTrees_depth_1_"+str(ntree), XTrain, YTrain, XTest, YTest)

		# print("Boosting depth 2")
		# model = GradientBoostingClassifier(n_estimators=ntree, max_depth = 2)
		# trainModel(model, "GBTrees_depth_2_"+str(ntree), XTrain, YTrain, XTest, YTest)

		print("Boosting depth 1")
		base = tree.DecisionTreeClassifier(max_depth=1)
		model = AdaBoostClassifier(base_estimator=base,n_estimators=5)
		trainModel(model, "GBTrees_depth_4_"+str(ntree), XTrain, YTrain, XTest, YTest)
		print(model.estimator_weights_)
		print(dir(model))
		# print(model.estimators_)
		# print(dir(model.estimators_))

if __name__ == "__main__":
   main(sys.argv[1:])
