import random
from abc import ABC
from abc import abstractmethod

import numpy as np

from .Tree import Node
from .Tree import Tree

class ClassificationTree(Tree):
	def __init__(self, min_splits):
		 super().__init__(min_splits)

	@abstractmethod
	def scoreNode(self,indices,node):
		pass

	@abstractmethod
	def generateSplit(self,f,indices):
		pass

	def generatePrediction(self,indices):
		classes = {}
		for y in self.Y[indices]:
			if y in classes:
				classes[y] += 1
			else:
				classes[y] = 1
		return max(classes, key=classes.get)

class ArchTree(ClassificationTree):
	def __init__(self,min_splits):
		self.min_splits = min_splits

	def scoreSplit(self,indices,node):
		left = []
		right = []
		for i in indices:
			if node.predict(self.X[i]):
				left.append(i)
			else:
				right.append(i)

		uleft = set(self.Y[left])
		uright = set(self.Y[right])

		if len(uleft) == len(self.Y[left]):
			return len(uleft)

		if len(uright) == len(self.Y[right]):
			return len(uright)

		# THIS SHOULD NOT HAPPEN!
		assert True

	def generateSplit(self,f,indices):
		node = Node()
		if self.isCategorical[f]:
			node.feature = f
			node.isCategorical = True
			node.leftChild = None
			node.rightChild = None
			node.prediction = None

			categories = set()
			for x in self.X[indices]:
				categories.add(x[f])

			splits = []
			for c in categories:
				node.split = c
				leftClass = None
				leftCnt = 0

				rightClass = None
				rightCnt = 0

				# TODO: CHANGE THIS
				for i in range(indices):
					if node.predict(self.X[i]):
						if leftClass is None:
							leftClass = self.Y[i]
						else:
							if leftCnt != None and leftClass == self.Y[i]:
								leftCnt += 1
							else:
								leftCnt = None
					else:
						if rightClass is None:
							rightClass = self.Y[i]
						else:
							if rightCnt != None and rightClass == self.Y[i]:
								rightCnt += 1
							else:
								rightCnt = None

					if leftClass is None and rightClass is None:
						continue

					if rightClass is None and leftClass is not None:
						splits.append((c,leftCnt))

					if leftClass is None and rightClass is not None:
						splits.append((c,rightCnt))

					if leftClass is not None and rightClass is not None:
						splits.append((c,leftCnt+rightCnt))

				if len(splits) == 0:
					return None
				else:
					maxC = splits[0][0]
					maxVal = splits[0][1]

					for c,val in splits[1:]:
						if val > maxVal:
							maxC = c
							maxVal = val

					node.split = c
					return node
		else:
			s_indices = sorted(s,key = lambda i: self.X[i,f])

			lower = s_indices[0]
			upper = s_indices[-1]
			
			if (lower == upper):
				return None

			ylower = self.Y[lower]
			xlower = self.Y[lower,f]
			lowercnt = 0
			for i in s_indices[1:]:
				if self.Y[i] == lower:
					lowercnt += 1
					xlower = self.X[i,f]
				else:
					break

			yupper = self.Y[upper]
			xupper = self.Y[upper,f]
			uppercnt = 0
			for i in reverse(s_indices[-1]):
				if self.Y[i] == lower:
					uppercnt += 1
					xupper = self.X[i,f]
				else:
					break

			if uppercnt > lowercnt:
				node.split = xupper
			else:
				node.split = xlower

			node.feature = f
			node.isCategorical = False
			node.leftChild = None
			node.rightChild = None
			node.prediction = None
		return node