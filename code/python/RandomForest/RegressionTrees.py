from abc import abstractmethod
import random

import numpy as np

from .Tree import Node
from .Tree import Tree

class RegressionTree(Tree):
	def __init__(self, min_splits):
		 super().__init__(min_splits)

	@abstractmethod
	def scoreSplit(self,indices,node):
		pass

	@abstractmethod
	def generateSplit(self,f,indices):
		pass

	def generatePrediction(self,indices):
		return np.mean(self.Y[indices])

class ExtraRegressionTree(RegressionTree):
	def __init__(self, min_splits=2):
		 super().__init__(min_splits)

	def scoreSplit(self,indices,node):
		# Variance Score
		left = []
		right = []
		for i in indices:
			if node.predict(self.X[i]):
				left.append(i)
			else:
				right.append(i)

		# mLeft = np.mean(self.Y[left])
		# mRight = np.mean(self.Y[right])

		# score = 0
		# for i in indices:
		# 	if node.predict(self.X[i]):
		# 		score += (self.Y[i] - mLeft)*(self.Y[i] - mLeft)
		# 	else:
		# 		score += (self.Y[i] - mRight)*(self.Y[i] - mRight)

		varAll = np.var(self.Y[indices])
		if (len(left) > 1):
			varLeft = len(left)/len(self.Y)*np.var(self.Y[left])
		else:
			varLeft = 0

		if (len(right) > 1):
			varRight = len(right)/len(self.Y)*np.var(self.Y[right])
		else:
			varRight = 0
		
		score = (varAll - varLeft - varRight) / varAll

		return left,right,score

	def generateSplit(self,f,indices):
		node = Node()
		if self.isCategorical[f]:
			categories = set()
			for x in self.X:
				categories.add(x[f])

			# choose random split point
			node.split = choice(categories)
			node.feature = f
			node.isCategorical = True
			node.leftChild = None
			node.rightChild = None
			node.prediction = None
		else:
			lower = min(self.X[indices,f])
			upper = max(self.X[indices,f])
			if (lower == upper):
				return None

			node.split = random.uniform(lower,upper)
			node.feature = f
			node.isCategorical = False
			node.leftChild = None
			node.rightChild = None
			node.prediction = None
		return node