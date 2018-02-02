import struct

import numpy as np

class TreeConverter:
	def __init__(self, dim, namespace, featureType):
		self.dim = dim
		self.namespace = namespace
		self.featureType = featureType

	def containsFloat(self,tree):
		for key in tree.nodes:
			if (isinstance(tree.nodes[key].split, float)):
				return True

		return False

	def getSplitRange(self,tree):
		splits = []
		for key in tree.nodes:
			if tree.nodes[key].prediction is None:
				splits.append(tree.nodes[key].split)

		if len(splits) == 0:
			return 0,0
		else:
			return min(splits), max(splits)

	def getDim(self):
		return self.dim

	def getNamespace(self):
		return self.namespace

	def getFeatureType(self):
		return self.featureType

	#def floatToHex(self, f):
		# Note: Use =I for unsigned int (see https://docs.python.org/2/library/struct.html#format-characters)
	#	return hex(struct.unpack('=i', struct.pack('=f', f))[0])

class ForestConverter:
	""" A ForestConverter converts an interal Forest structure to
		architectural specific forest code. To do so, it uses a
		treeConverter to convert single trees into appropriate
		c-code and adds some additional glue-code for prediction
	"""
	def __init__(self, treeConverter):
		""" Generate a new ForestConverter

		Args:
			treeConverter: A tree converter
		"""
		assert(issubclass(type(treeConverter), TreeConverter))
		self.treeConverter = treeConverter

	def getCode(self, forest, numClasses):
		""" Generate the actual code for the given forest

		Args:
			forest (TYPE): The forest object

		Returns:
			Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
			a *.h file and cppCode contains the code (=string) for a *.cpp file
		"""
		dim = self.treeConverter.getDim()
		namespace = self.treeConverter.getNamespace()
		featureType = self.treeConverter.getFeatureType()

		headerCode = "unsigned int {namespace}_predict({feature_t} const pX[{dim}]);\n".replace("{dim}", str(dim)).replace("{namespace}", namespace).replace("{feature_t}", featureType)
		cppCode = "unsigned int {namespace}_predict({feature_t} const pX[{dim}]) {\n".replace("{dim}", str(dim)).replace("{namespace}", namespace).replace("{feature_t}", featureType)

		initCode = "{"
		for i in range(0,numClasses):
			initCode += "0,"
		initCode = initCode[:-1] + "};\n"

		cppCode += "	unsigned int predCnt[{num_classes}] = " + initCode
		for i in range(len(forest.trees)):
			cppCode += "	predCnt[{namespace}_predict{id}(pX)]++;\n".replace("{id}", str(i)).replace("{namespace}", namespace)
		cppCode += """unsigned int pred = 0;
				unsigned int cnt = predCnt[0];
				for (unsigned int i = 1; i < {num_classes}; ++i) {
					if (predCnt[i] > cnt) {
						cnt = predCnt[i];
						pred = i;
					}
				}
				return pred;
			}\n"""
		cppCode = cppCode.replace("{num_classes}", str(numClasses))

		for i in range(len(forest.trees)):
			tHeader, tCode = self.treeConverter.getCode(forest.trees[i], i)
			headerCode += tHeader
			cppCode += tCode

		return headerCode, cppCode
