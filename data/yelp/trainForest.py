#!/usr/bin/env python3

import json,sys
import os.path
import timeit
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfTransformer

sys.path.append('../../code/python')

import sys
sys.setrecursionlimit(20000)

from RandomForest import RandomForest

def main(argv):
	print("Loading businesses")
	businesses = {}
	f = open("dataset/business.json",'r')
	for line in f:
		jsonEntry = json.loads(line.strip())
		businesses[jsonEntry['business_id']] = jsonEntry['categories']

	print("Loading reviews")
	texts = []
	Y = []

	f = open("dataset/review.json",'r')
	for line in f:
		jsonEntry = json.loads(line.strip())

		if "Restaurants" in businesses[jsonEntry['business_id']] and (jsonEntry['stars']==1 or jsonEntry['stars']==5):
			texts.append(jsonEntry['text'])
			Y.append(jsonEntry['stars'])

	count_vect = CountVectorizer(stop_words='english', lowercase=True,max_df=0.1, min_df=0.0001)
	X = count_vect.fit_transform(texts)

	print("Data shape:",X.shape)
	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)
	# NOTE: WE take a sample here which can be saved on the disk
	indices = np.random.choice(range(XTest.shape[0]),size=25000)
	XTest = XTest[0:25000].todense()
	YTest = YTest[0:25000]

	NTrees = [1,50]
	for ntree in NTrees:
		clf = RandomForestClassifier(n_estimators=ntree, n_jobs=8) 
		print("Fitting model on " + str(XTrain.shape[0]) + " data points")
		#clf.fit(XTrain,YTrain)
		
		print("Testing model on " + str(XTest.shape[0]) + " data points")
		start = timeit.default_timer()
		#YPredicted = clf.predict(XTest)
		end = timeit.default_timer()
		#print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
		#print("Accuracy:%s" % accuracy_score(YTest, YPredicted))
		print("Total time: " + str(end - start) + " ms")
		print("Throughput: " + str(XTest.shape[0] / (float(end - start)*1000)) + " #elem/ms")

		#print("Saving model to JSON on disk")
		#forest = RandomForest.RandomForestClassifier(None)
		#forest.fromSKLearn(clf)

		#with open("text/forest_"+str(ntree)+".json",'w') as outFile:
		#	outFile.write(forest.str())

		with open("text/forest_"+str(ntree)+"_test.csv", 'w') as outFile:
			for x,y in zip(XTest, YTest):
				line = str(y)
				print("x=",x[0][0][0][0])
				print("x.s=",x.shape)
				for xi in x[0][0]:
					line += "," + str(xi)
				outFile.write(line + "\n")

		print("*** Summary ***")
		print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
		print(str(X.shape[0]) + "\t" + str(XTest.shape[1]) + "\t" + str(accuracy_score(YTest, YPredicted)) + "\t" + str(forest.getAvgDepth()))

if __name__ == "__main__":
   main(sys.argv[1:])
