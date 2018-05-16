#!/usr/bin/env python3

import sys
import numpy as np
import pprint, json

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# TODO MULTICLASS

import Tree
import Forest

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

def testModel(X,Y,m, isTree = False):
	m.fit(X,Y)

	if isTree:
		mymodel = Tree.Tree()
	else:
		mymodel = Forest.Forest()

	mymodel.fromSKLearn(m)

	for (x,y) in zip(X,Y):
		skpred = m.predict(x.reshape(1, -1))[0]
		dnode,mypred = mymodel.predict(x)
		mypred = mypred.argmax()

		if (skpred != mypred):
			print(m.predict_proba(x.reshape(1, -1)))
			print(m.predict(x.reshape(1, -1)))
			print(mymodel.predict(x))
			print(mymodel.pstr())

			return False
	return True

def main(argv):
	data = load_breast_cancer()
	X = data.data.astype(dtype=np.float32)
	Y = data.target

	print("BINARY CLASSIFICATION TEST")
	print("### Decision Tree ###")
	if testModel(X,Y,DecisionTreeClassifier(), True):
		print("    test passed")

	print("### Extra Tree ###")
	if testModel(X,Y,ExtraTreesClassifier(n_estimators=20)):
		print("    test passed")

	print("### Random Forest ###")
	if testModel(X,Y,RandomForestClassifier(n_estimators=20)):
		print("    test passed")

	print("### AdaBoost Classifier ###")
	if testModel(X,Y,AdaBoostClassifier(n_estimators=20)):
		print("    test passed")

	print()
	print()
	print("MULTICLASS CLASSIFICATION TEST")
	data = load_iris()
	X = data.data.astype(dtype=np.float32)
	Y = data.target
	print("### Decision Tree ###")
	if testModel(X,Y,DecisionTreeClassifier(), True):
		print("    test passed")

	print("### Extra Tree ###")
	if testModel(X,Y,ExtraTreesClassifier(n_estimators=20)):
		print("    test passed")

	print("### Random Forest ###")
	if testModel(X,Y,RandomForestClassifier(n_estimators=20)):
		print("    test passed")

	print("### AdaBoost Classifier ###")
	if testModel(X,Y,AdaBoostClassifier(n_estimators=20)):
		print("    test passed")


if __name__ == "__main__":
   main(sys.argv[1:])
