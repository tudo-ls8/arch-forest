import numpy as np
import json
from sklearn.tree import _tree

from . import Tree

class Forest:
	def __init__(self):
		self.trees = []

	def fromSKLearn(self,forest,roundSplit = False):
		for e in forest.estimators_:
			tree = self.generateTree()
			tree.fromSKLearn(e,roundSplit)
		
			self.trees.append(tree)
		
	def fromJSON(self, jsonFile):
		with open(jsonFile) as data_file:    
			data = json.load(data_file)

		for x in data:
			tree = self.generateTree()
			tree.fromJSON(x)

			self.trees.append(tree)

	def str(self):
		s = "["
		for tree in self.trees:
			s += tree.str() + ","
		s = s[:-1] + "]"

		return s

	def generateTree(self):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")

	def predict(self):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")
		
	## SOME STATISTICS FUNCTIONS ##

	def getSubTrees(self, minProb, maxNumNodes):
		subTrees = []
		for t in self.trees:
			subTree, prob, size = t.getSubTree(minProb,maxNumNodes)
			subTrees.append(subTree)
		return subTrees

	# def getAvgProb(self):
	# 	return sum([t.getAvgProb() for t in self.trees]) / len(self.trees)

	# def getMaxProb(self, n_top = 1):
	# 	sums = np.array([0 for i in range(n_top)])
	# 	for t in self.trees:
	# 		sums = np.add(sums,t.getMaxProb(n_top))

	# 	return sums / len(self.trees)

	def getAvgDepth(self):
		return sum([t.getAvgDepth() for t in self.trees]) / len(self.trees)

	# def getMaxDepth(self):
	# 	return sum([t.getMaxDepth() for t in self.trees]) / len(self.trees)

	# def getAvgNumNodes(self):
	# 	return sum([t.getNumNodes() for t in self.trees]) / len(self.trees)

	# def getAvgNumLeafs(self):
	# 	return sum([t.getNumLeaf() for t in self.trees]) / len(self.trees)

	# def getMaxNumLeafs(self):
	# 	return max([t.getNumLeaf() for t in self.trees])

	# def getMinNumLeafs(self):
	# 	return min([t.getNumLeaf() for t in self.trees])

	# def getAvgNumPaths(self):
	# 	return sum([t.getNumLeaf() for t in self.trees]) / len(self.trees)