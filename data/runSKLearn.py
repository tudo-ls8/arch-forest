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

def readFile(path):
	f = open(path, 'r')
	X = []
	Y = []
	for row in f:
		entries = row.strip().split(",")
		x = [float(e) for e in entries[1:]]
		y = float(entries[0])
		X.append(x)
		Y.append(y)

	return np.array(X), np.array(Y)

def main(argv):
	if len(argv)<1:
		print("Please give a sub-folder / dataset to be used")
		return
	else:
		basepath = argv[0].strip("/")

	#print("Reading test file")
	XTest,YTest = readFile(basepath + "/test.csv")

	for f in sorted(os.listdir(basepath + "/text/")):
		if f.endswith(".pkl"): 
			#print("Loading model", f)
			clf = joblib.load(basepath + "/text/" + f)

			# Burn in phase
			for i in range(2):
				YPredicted = clf.predict(XTest)
			
			#print("Accuracy:%s" % accuracy_score(YTest, YPredicted))	
	
			# Actual measure
			runtimes = []
			for i in range(20):
				start = timeit.default_timer()
				YPredicted = clf.predict(XTest)
				end = timeit.default_timer()
				runtimes.append((end-start)*1000)	

			print(basepath+"/text/"+f,",",np.mean(runtimes),",",np.var(runtimes))			

if __name__ == "__main__":
   main(sys.argv[1:])