import json
from functools import reduce

import numpy as np

from sklearn.tree import _tree

from . import Node

class Tree():
	def __init__(self, min_splits = 2):
		# For simpler computations of statistics, we will also save all nodes in a
		# dictionary where (key = nodeID, value = actuale node)
		self.nodes = {}
		self.min_splits = min_splits

		# Pointer to the root node of this tree
		self.head = None

	def scoreSplit(self,indices,node):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")

	def generatePrediction(self,indices):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")

	def generateSplit(self,f,indices):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")

	def generateNode(self,indices):
		if len(set(self.Y[indices])) == 1 or len(indices) < self.min_splits:
			node = Node.Node()
			node.prediction = self.generatePrediction(indices)
			return [],[],node
		else:
			scores = []
			datasplits = []
			nodes = []
			for f in range(len(self.X[0])):
				node = self.generateSplit(f,indices)
				if node is None:
					continue

				node.isCategorical = self.isCategorical[f]

				nodes.append(node)
				leftI,rightI,score = self.scoreSplit(indices, node)
				scores.append(score)
				datasplits.append((leftI,rightI))

			if len(scores) == 0:
				node.prediction = self.generatePrediction(indices)

				return [],[],node
			else:
				bestNode = scores.index(max(scores))
				print("best score:", max(scores))
				print("len(left):", len(datasplits[bestNode][0]))
				print("len(right):", len(datasplits[bestNode][1]))
				return datasplits[bestNode][0],datasplits[bestNode][1],nodes[bestNode]

	def fit(self,X,Y):
		assert len(X) > 0
		assert len(Y) > 0
		#print("Training tree", self.id)

		self.isCategorical = []

		# Check for types of features
		for xi in X[0]:
			self.isCategorical.append(isinstance(xi,str))
		
		# For splits only use indices
		self.X = X
		self.Y = Y

		datasplits = [range(len(Y))] 

		parents = []
		while len(datasplits) > 0:
			indices = datasplits.pop(0)
			left,right,node = self.generateNode(indices)
			nodeIndex = len(self.nodes)
			node.id = nodeIndex

			if (len(parents) > 0):
				parent = parents.pop(0)
				if self.nodes[parent].leftChild is None:
					self.nodes[parent].leftChild = node #nodeIndex
				else:
					self.nodes[parent].rightChild = node #nodeIndex

			if node.prediction is None:
				node.probLeft = len(left) / len(indices)
				node.probRight = len(right) / len(indices)
				node.numSamples = len(indices)

				datasplits.append(left)
				parents.append(nodeIndex)

				datasplits.append(right)
				parents.append(nodeIndex)
			
			if self.head is None:
				self.head = node

			self.nodes[nodeIndex] = node
			
	def predict(self,X):
		YPred = []
		for x in X:
			curNode = self.nodes[0]
			while curNode.prediction is None:
				if curNode.predict(x):
					curNode = curNode.leftChild
				else:
					curNode = curNode.rightChild
			YPred.append(curNode.prediction)

		return YPred

	def fromTree(self, nodes, head, min_splits=2):
		self.nodes = nodes
		self.head = head
		self.min_splits = min_splits

	def fromJSON(self, json, first = True):
		node = Node.Node()
		node.fromJSON(json)
		self.nodes[node.id] = node
		if node.prediction is None:
			node.rightChild = self.fromJSON(json["rightChild"], False)
			node.leftChild = self.fromJSON(json["leftChild"], False)

		if first:
			self.head = node

		return node

	def fromSKLearn(self, tree, roundSplit = False):
		self.head = self._fromSKLearn(tree.tree_, 0, tree.classes_, roundSplit)

	def _fromSKLearn(self, tree, curNode, classes, roundSplit = False):
		""" Loads a tree from sci-kit internal data structure into this object
		
		Args:
		    tree (TYPE): The sci-kit tree
		    curNode (int, optional): The current node index (default = 0 ==> root node of the tree)
		
		Returns:
		    TYPE: The root node of the extracted tree structure
		"""
		node = Node.Node()
		node.fromSKLearn(tree, curNode, classes, roundSplit)
		node.id = len(self.nodes)
		self.nodes[node.id] = node

		if node.prediction is None:
			leftChild = tree.children_left[curNode]
			rightChild = tree.children_right[curNode]
			
			node.leftChild = self._fromSKLearn(tree, leftChild, classes, roundSplit)
			node.rightChild = self._fromSKLearn(tree, rightChild, classes, roundSplit)

		return node

	def str(self, head = None):
		if head is None:
			head = self.head

		if head.prediction is not None:
			return head.str()
		else:
			leftChilds = self.str(head.leftChild)
			rightChilds = self.str(head.rightChild)
			s = head.str(leftChilds, rightChilds)
			return s

	## SOME STATISTICS FUNCTIONS ##
	def getSubTree(self, minProb, maxNumNodes):
		allSubPaths = self.getAllPaths()
		allSubPaths.sort(key = lambda x : len(x), reverse=True)
		paths = []
		curProb = 1.0
		curSize = 0

		added = True
		while(added):
			added = False
			for p in allSubPaths:
				prob = np.prod([n[1] for n in p])

				if curProb + prob > minProb and curSize + len(p) < maxNumNodes:
					paths.append(p)
					curSize += len(p)
					curProb += prob

					added = True

					# Does this work during iteration?!
					break
			
			if (added):
				allSubPaths.remove(paths[-1])

		return paths,curProb,curSize

	def getProbAllPaths(self, node = None, curPath = [], allPaths = [], pathNodes = [], pathLabels = []):
		if node is None:
			node = self.head

		if node.prediction is not None:
			allPaths.append(curPath)
			pathLabels.append(pathNodes)
			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Leaf nodes "+str(node.id)+" : "+str(curProb))
		else:
			try:
				pathNodes.index(node.id)
				#this node is root
			except ValueError:
				pathNodes.append(node.id)
				curPath.append(1)

			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Root or Split nodes "+str(node.id)+ " : " +str(curProb))
			self.getProbAllPaths(node.leftChild, curPath + [node.probLeft], allPaths, pathNodes + [node.leftChild.id], pathLabels)
			self.getProbAllPaths(node.rightChild, curPath + [node.probRight], allPaths, pathNodes + [node.rightChild.id], pathLabels)

		return allPaths, pathLabels

	# def getNumNodes(self):
	# 	return len(self.nodes)

	# def getMaxDepth(self):
	# 	paths = self.getAllPaths()
	# 	return max([len(p) for p in paths])

	def getAvgDepth(self):
		paths = self.getAllPaths()
		return sum([len(p) for p in paths]) / len(paths)
	
	# This returns all sub-paths starting with the root node
	def getAllPaths(self, node = None, curPath = [], allPaths = None):
		# NOTE: FOR SOME REASON allPaths = [] DOES NOT CORRECTLY RESET
		# 		THE allPaths VARIABLE. THUS WE USE NONE HERE 
		if allPaths is None:
			allPaths = []

		if node is None:
			node = self.head

		if node.prediction is not None:
			curPath.append((node.id,1))
			allPaths.append(curPath)
		else:
			# We wish to include all sub-paths, not only complete paths from root to leaf node
			if len(curPath) > 0:
				allPaths.append(curPath)

			self.getAllPaths(node.leftChild, curPath + [(node.id,node.probLeft)], allPaths)
			self.getAllPaths(node.rightChild, curPath + [(node.id,node.probRight)], allPaths)
		
		return allPaths

	# def getMaxProb(self, top_n = 1):
	# 	paths = self.getAllPaths()
	# 	probs = [reduce(lambda x, y: x*y, path) for path in paths]
	# 	probs.sort(reverse=True)

	# 	return probs[0:top_n]

	# def getAvgProb(self):
	# 	paths = self.getAllPaths()
	# 	return sum( [reduce(lambda x, y: x*y, path) for path in paths] ) / len(paths)