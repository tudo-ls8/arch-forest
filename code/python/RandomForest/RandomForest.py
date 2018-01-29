import concurrent.futures as cf
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from multiprocessing import Pool
from sklearn.tree import _tree

import numpy as np

from .ClassificationTrees import ArchTree
from RandomForest.RegressionTrees import ExtraRegressionTree

from .Forest import Forest

class RandomForest(Forest):
	def __init__(self, numTrees, bootstrap_sample = False, min_splits = 2):
		super().__init__()
		self.numTrees = numTrees
		self.bootstrap_sample = bootstrap_sample
		self.min_splits = min_splits

	def _fit(self,indices):
		tree = self.generateTree(self.min_splits)
		tree.fit(self.X[indices],self.Y[indices])
		return tree

	def fit(self,X,Y):
		self.X = X
		self.Y = Y
		indices = []
		for i in range(0, self.numTrees):
			if self.bootstrap_sample:
				indices.append(np.random.choice(len(X), len(X), replace=True))
			else:
				indices.append([i for i in range(len(X))])

		# do the actual training
		pool = Pool(4) 
		self.trees = pool.map(self._fit, indices)

	def predict(self,X):
		raise NotImplementedError("This function should not be called directly, but only by a sub-class")

class RandomForestClassifier(RandomForest):
	def generateTree(self):
		return ExtraRegressionTree(self.min_splits)

	def predict(self,X):
		YPred = []
		for x in X:
			classes = {}
			for t in self.trees:
				ypred = t.predict([x])[0]
				if ypred in classes:
					classes[ypred] += 1
				else:
					classes[ypred] = 1
			ypred = max(classes, key=classes.get)
			YPred.append(ypred)
		return YPred

class RandomForestRegressor(RandomForest):
	def generateTree(self):
		return ArchTree(self.min_splits)

	def predict(self,X):
		YPred = []
		for x in X:
			y = 0
			for t in self.trees:
				y += t.predict([x])[0]
			YPred.append(y / self.numTrees)
		return YPred