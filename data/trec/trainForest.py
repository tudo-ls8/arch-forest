#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np
import os.path
import json
import timeit,codecs

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

sys.path.append('../../code/python')
from RandomForest import RandomForest

def readFile(path):
	#try:
	f = codecs.open(path, 'r','utf-8',errors="ignore")
	text = ""
	for row in f:
		text += row

	return text
	# except Exception as e:
	# 	return ""

def main(argv):
	outPath = "./text"
	
	texts = []
	Y = []
	f = open("trec07p/full/index", 'r')
	for row in f:
		entries = row.replace("\n","").split(" ")
		tmp = readFile("trec07p/" + entries[1].split("..")[1])
		if len(tmp) > 0:
			texts.append(tmp)
			Y.append(1 if entries[0] == "spam" else 0)
		
	
	count_vect = CountVectorizer(stop_words='english', lowercase=True,max_df=0.1, min_df=0.001)
	X = count_vect.fit_transform(texts)
	Y = np.array(Y)

	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)

	with open("test.csv") as outFile:
		for i in range(XTest.shape[0]):
	 		line = str(YTest[i])
	 		for j in range(XTest.shape[1]):
	 			line += "," + str(XTest[i,j])
	 		outFile.write(line + "\n")

	NTrees = [25]
	for ntree in NTrees:
		clf = RandomForestClassifier(n_estimators=ntree, n_jobs=8) 
		print("Fitting model on " + str(XTrain.shape[0]) + " data points")
		clf.fit(XTrain,YTrain)
		
		print("Testing model on " + str(XTest.shape[0]) + " data points")
		start = timeit.default_timer()
		YPredicted = clf.predict(XTest)
		end = timeit.default_timer()
		print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
		print("Accuracy:%s" % accuracy_score(YTest, YPredicted))
		print("Total time: " + str(end - start) + " ms")
		print("Throughput: " + str(XTest.shape[0] / (float(end - start)*1000)) + " #elem/ms")

		print("Saving model to JSON on disk")
		forest = RandomForest.RandomForestClassifier()
		forest.fromSKLearn(clf, True)
		
		if not os.path.exists("text"):
			os.makedirs("text")

		with open("text/forest_"+str(ntree)+".json",'w') as outFile:
			outFile.write(forest.str())
		
		print("Saving model to PKL on disk")
		joblib.dump(clf, "text/forest_"+str(ntree)+".pkl")

		print("*** Summary ***")
		print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
		print(str(XTest.shape[0]) + "\t" + str(XTest.shape[1]) + "\t" + str(accuracy_score(YTest, YPredicted)) + "\t" + str(forest.getAvgDepth()))

if __name__ == "__main__":
   main(sys.argv[1:])
