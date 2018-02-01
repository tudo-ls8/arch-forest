#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json, random
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
import timeit
from sklearn.ensemble import VotingClassifier

sys.path.append('../../code/python')

from RandomForest import Forest
from RandomForest import Tree
from RandomForest import Node
from RandomForest import RandomForest

# def generateVeryDifficultData(N, dim):
# 	X = []
# 	Y = []
# 	for i in range(0, N):
# 		x = np.random.randn(dim)
# 		x[0] = i;
# 		X.append(x)
# 		Y.append(bool(x[0] % 2 == 0))

# 	X = np.array(X)
# 	Y = np.array(Y)
# 	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)

# 	return XTrain,XTest,YTrain,YTest

# def generateDifficultData(N, dim):
# 	beta = np.random.normal(scale = 10, size = dim)
# 	print(beta)
# 	# beta = [-1 if i % 2 == 0 else 1 for i in range(dim)]

# 	# # NOTE: If dim is odd, we have to match the entries so that the
# 	# # dot product of np.dot([1 1 1 ... 1], beta) == 0
# 	# if dim % 2 != 0:
# 	# 	beta[0] = 0.5
# 	# 	beta[1] = 0.5
# 	# basis = [1 for i in range(dim)]

# 	X = []
# 	Y = []
# 	for i in range(N):
# 		x = np.random.randint(low = 0, high = 2*N, size=dim) - N
# 		X.append(x)
# 		Y.append(bool(np.dot(beta, x) <= 0))

# 	return np.array(X), np.array(Y)

# def generateEasyData(N, dim):
# 	X = []
# 	Y = []
# 	for i in range(N):
# 		x = [i for i in range(0,dim)]
# 		x[0] = i
# 		X.append(x)
# 		Y.append(bool(i > N/2))

# 	X = np.array(X)
# 	Y = np.array(Y)
# 	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)

# 	return XTrain,XTest,YTrain,YTest

def generateChain(N, numNodes, pathProb):
	head = None
	tmpNode = None

	probRight = pathProb**(1/numNodes)

	X = []
	Y = []

	nodes = {}
	lower = 0
	upper = 1
	curLabel = True
	idcnt = 0
	for nodecnt in range(numNodes):
		if head is None:
			head = Node.Node()
			tmpNode = head

		NLeft = int(N*(1-probRight))
		if NLeft == 0:
			NLeft = 1

		tmpNode.probLeft = 1-probRight
		tmpNode.probRight = probRight
		tmpNode.numSamples = N
		tmpNode.isCategorical = False
		tmpNode.feature = 0
		tmpNode.split = upper
		tmpNode.id = idcnt
		nodes[idcnt] = tmpNode
		idcnt += 1

		tmpNode.leftChild = Node.Node()
		tmpNode.leftChild.id = idcnt
		nodes[idcnt] = tmpNode.leftChild
		idcnt += 1

		tmpNode.leftChild.prediction = curLabel
		tmpNode.leftChild.numSamples = NLeft
		tmpNode.rightChild = Node.Node()
		tmpNode = tmpNode.rightChild

		for i in range(NLeft):
			x = random.uniform(lower, upper)
			X.append([x])
			Y.append(curLabel)
		
		N -= NLeft
		assert N > 0

		curLabel = not curLabel
		lower = upper
		upper = upper + 1

	nodes[idcnt] = tmpNode
	tmpNode.prediction = curLabel
	tmpNode.id = idcnt
	tmpNode.numSamples = N

	for i in range(N):
		x = random.uniform(lower, upper)
		X.append([x])
		Y.append(curLabel)

	tTree = Tree.Tree()
	tTree.fromTree(nodes, head)
	# print(tTree.str())

	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)

	return XTrain,XTest,YTrain,YTest,tTree

def testTree(tree,XTest,YTest):
	print("\tPerform prediction on test data with " + str(len(XTest)) + " data points")
	YPredicted = tree.predict(XTest)
	#print("\tConfusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
	print("\tAccuracy:%s" % accuracy_score(YTest, YPredicted))

	print("\tPerform speed test")
	start = timeit.default_timer()
	YPredicted_ = tree.predict(XTest)
	end = timeit.default_timer()
	print("\tThroughput: " + str(len(YTest) / (float(end - start)*1000)) + " #elem/ms")	

def trainTree(XTrain,YTrain):
	NTree = 25
	#clf = DecisionTreeClassifier(max_features = None, max_depth = None,min_samples_split=2)
	clf = RandomForestClassifier(NTree,max_features = None, max_depth = None,min_samples_split=2)

	print("\tFitting model on " + str(len(XTrain)) + " data points")
	clf.fit(XTrain,YTrain)

	print("\tPerform prediction on training data with " + str(len(XTrain)) + " data points")
	print("\t\t Note: This should give an accuracy of 1! If not, something is wrong.")
	YPredicted = clf.predict(XTrain)
	#print("\tConfusion matrix:\n%s" % confusion_matrix(YTrain, YPredicted))
	print("\tAccuracy:%s" % accuracy_score(YTrain, YPredicted))

	return clf

def writeToFile(filename, XTrain,XTest,YTrain,YTest):
	with open(filename + "_train.csv",'w') as train_file:
		for x,y in zip(XTrain, YTrain):
			line = ("1" if y else "0")
			for xi in x:
				line += "," + str(xi)
			line += "\n"
			train_file.write(line)

	with open(filename + "_test.csv",'w') as test_file:
		for x,y in zip(XTest, YTest):
			line = ("1" if y else "0")
			for xi in x:
				line += "," + str(xi)
			line += "\n"
			test_file.write(line)

def main(argv):
	# Parameter
	outPath = "./text"

	# The dimension of the feature vector (=length of the feature vectors)
	#dim = 10
	# Number of training data to be generated
	N = 10000

	probs = [0.6,0.7,0.8,0.9,0.95,0.99]
	sizes = [5,10,15,20,25,30,35,40,45,50]

	for p in probs:
		for s in sizes:
			print("### Generating decision chain ###")
			print("\tp =", p, " s =", s)
			XTrain,XTest,YTrain,YTest,mychain = generateChain(N, numNodes = s, pathProb = p)
			skchain  = trainTree(XTrain,YTrain)
			testTree(skchain,XTest,YTest)
			filename = outPath + "/" + "p_" + str(p) + "_s_" + str(s)
			writeToFile(filename,XTrain,XTest,YTrain,YTest)
			
			tChain = RandomForest.RandomForestClassifier(None)
			tChain.fromSKLearn(skchain)
			# with open(outPath + "/" + "skchain.json",'w') as tree_file:
			# 	tree_file.write(tChain.str())

			with open(filename + ".json",'w') as tree_file:
				tree_file.write(tChain.str())

			print("")
	# print("### Generating very difficult training data ###")
	# XTrain,XTest,YTrain,YTest = generateVeryDifficultData(N, dim);
	# veryDifficultTree = trainTree(XTrain,YTrain)
	# testTree(veryDifficultTree,XTest,YTest)
	# writeToFile(outPath + "/" + "difficult_tree",XTrain,XTest,YTrain,YTest)

	# # Load from SKLearn
	# tree = Tree.Tree()
	# tree.fromSKLearn(veryDifficultTree)
	# print("Test fromSKLearn-Tree")
	# testTree(tree,XTest,YTest)

	# # write to JSON
	# with open(outPath + "/" + "difficult_tree.json",'w') as tree_file:
	# 	tree_file.write(tree.str())

	# # Load from JSON
	# loadedTree = Tree.Tree()
	# with open(outPath + "/" + "difficult_tree.json") as data_file:
	# 	data = json.load(data_file)

	# loadedTree.fromJSON(data)
	# print("Test fromJSON-Tree")
	# testTree(loadedTree,XTest,YTest)

	# print("\n\n### Generating easy training data ###")
	# XTrain,XTest,YTrain,YTest = generateEasyData(N, dim);
	# easyTree = trainTree(XTrain,YTrain)
	# testTree(easyTree,XTest,YTest)
	# writeToFile(outPath + "/" + "easy_tree",XTrain,XTest,YTrain,YTest)

	# #Load from SKLearn
	# tree = Tree.Tree()
	# tree.fromSKLearn(easyTree)
	# print("Test fromSKLearn-Tree")
	# testTree(tree,XTest,YTest)

	# # write to JSON
	# with open(outPath + "/" + "easy_tree.json",'w') as tree_file:
	# 	tree_file.write(tree.str())

	# # Load from JSON
	# loadedTree = Tree.Tree()
	# with open(outPath + "/" + "easy_tree.json") as data_file:
	# 	data = json.load(data_file)
	# loadedTree.fromJSON(data)

	# loadedTree.fromJSON(data)
	# print("Test fromJSON-Tree")
	# testTree(loadedTree,XTest,YTest)
	#Do something with loadedTree

if __name__ == "__main__":
   main(sys.argv[1:])
