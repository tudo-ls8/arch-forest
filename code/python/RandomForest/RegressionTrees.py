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
		node = Node.Node()
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

class GradientRegressionTree(RegressionTree):
	def __init__(self, min_splits=2):
		 super().__init__(min_splits)
		 self.losses = {}

	def scoreSplit(self,indices,node):
		# Variance Score
		left = []
		right = []
		for i in indices:
			if node.predict(self.X[i]):
				left.append(i)
			else:
				right.append(i)

		score = 1.0 / self.losses[node.feature]
		
		return left,right,score

	def generateSplit(self,f,indices):
		node = Node.Node()
		if self.isCategorical[f]:
			raise NotImplementedError("To be done")
		else:
			l = 0
			u = 1
			s = 0.5
			mu = 0.02
			NSteps = 10
			avgLoss = 0
			#print("Feature f:", f)
			for i in range(NSteps):
				j = np.random.choice(indices)
				x = self.X[j]
				y = self.Y[j]
				pred = l+u/(1+np.exp(-x[f]+s))
				loss = (pred - y)*(pred - y)
				#print("\tpred =", pred)
				#print("\tloss =", loss)

				avgLoss += loss

				# Perform SGD
				l = l - mu*2*(pred - loss)
				u = u - mu*2*(pred - loss)*(1/(1+np.exp(-x[f]+s)))
				s = s - mu*2*(pred - loss)*(pred)*(1-pred)
				#print("\tl =",l)
				#print("\tu =",u)
				#print("\ts =",s)
				#print("---\n")

			#print("avgLoss=",avgLoss)
			self.losses[f] = avgLoss
			node.split = s
			node.feature = f
			node.isCategorical = False
			node.leftChild = None
			node.rightChild = None
			node.prediction = None
		return node