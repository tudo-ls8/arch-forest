#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
import timeit
from sklearn.ensemble import VotingClassifier

from Forest import *

def writeToFile(filename, XTrain,XTest,YTrain,YTest):
	#with open(filename + "_TRAIN.csv",'w') as train_file:
	#	for x,y in zip(XTrain, YTrain):
	#		line = ("1" if y else "0")
	#		for xi in x:
	#			line += "," + str(xi)
	#		line += "\n"
	#		train_file.write(line)

	with open(filename + "_TEST.csv",'w') as test_file:
		for x,y in zip(XTest, YTest):
			line = ("1" if y else "0")
			for xi in x:
				line += "," + str(xi)
			line += "\n"
			test_file.write(line)

def generateVeryDifficultData(N, dim):
	X = []
	Y = []
	for i in range(0, N):
		x = [i for i in range(0,dim)]
		x[0] = i;
		X.append(x)
		Y.append(bool(x[0] % 2 == 0))

	X = np.array(X)
	Y = np.array(Y)
	XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.25)

	return XTrain,XTest,YTrain,YTest

def generateDifficultTree(numSamples, threshold, depth, nodeID):
	threshold=int(threshold)
	if(threshold % 2 == 0):
		jsonFile = """{"id":{id},"type":"split","numSamples":{numSamples},"negProb":0.5,"posProb":0.5,"compare":"<=","feature":0,"threshold":{threshold},"leftChild":{leftChild},"rightChild":{rightChild}}"""
		jsonFile = jsonFile.replace("{id}", str(nodeID)).replace("{numSamples}", str(numSamples)).replace("{threshold}", str(threshold-1)).replace("{leftChild}", generateDifficultTree(numSamples/2, threshold-(numSamples/4), depth+1, 2*nodeID+2)).replace("{rightChild}", generateDifficultTree(numSamples/2, (numSamples/4)+threshold, depth+1, 2*nodeID+1))
	else:
		jsonFile = """{"id":{id},"type":"split","numSamples":{numSamples},"negProb":0.5,"posProb":0.5,"compare":"<=","feature":0,"threshold":{threshold},"leftChild":{"id":{idL},"type":"leaf","numSamples":1,"negProb":0.0,"posProb":1.0},"rightChild":{"id":{idR},"type":"leaf","numSamples":1,"negProb":1.0,"posProb":0.0}"""+"}"
		jsonFile = jsonFile.replace("{id}", str(nodeID)).replace("{numSamples}", str(numSamples)).replace("{threshold}", str(threshold-1)).replace("{idL}", str(2*nodeID+2)).replace("{idR}", str(2*nodeID+1))

	return jsonFile

def main(argv):
	if(len(argv)<1):
		print("Please give the depth of the tree")

	# Parameter
	outPath = "../../data/"
	
	# The depth of the generated balanced tree
	depth = int(argv[0])
	
	# The dimension of the feature vector (=length of the feature vectors)
	dim = 1
	
	numSamples = 2**depth

	XTrain,XTest,YTrain,YTest = generateVeryDifficultData(numSamples, dim)
	writeToFile(outPath + "/" + "difficult_tree",XTrain,XTest,YTrain,YTest)
	jsonFile = generateDifficultTree(numSamples, numSamples/2, depth, 0)
	with open(outPath + "/" + "difficult_tree.json",'w') as tree_file:
		tree_file.write(jsonFile)

main(sys.argv[1:])
