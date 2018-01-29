from ForestConverter import TreeConverter
import numpy as np

class DNFConverter(TreeConverter):
	""" A DNFConvter converts a DecisionTree into its disjunctive normal form in c language
	"""
	def __init__(self, dim, namespace, featureType):
		super().__init__(dim, namespace, featureType)

	def getComparisonType(self):
		raise NotImplementedError("getComparisonType not implemented! Did you use super class?")

	def getImplementation(self, treeID, tree):
		featureType = self.getFeatureType()
		headerCode = "bool {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
						.replace("{treeID}", str(treeID)) \
						.replace("{dim}", str(self.dim)) \
						.replace("{namespace}", self.namespace) \
						.replace("{feature_t}", featureType)
		code = ""
		comparators = {}

		nodeID = 0
		for key, node in tree.nodes.items():
			if not node.type == "leaf":
				comparators["(pX[" + str(node.feature) + "] <= " + str(node.threshold) + ");\n"] = nodeID
				nodeID += 1

		for key, c in comparators.items():
			code += self.getComparisonType() + " c" + str(c) + " = " + key;
			

		def _treeToDNF(node, curDNFStr, curConjunction):
			if node.type == "leaf":
				if node.posProb > node.negProb:
					curDNFStr += "(" + curConjunction[:-2] + ")||"

				return curDNFStr
			else:
				compID = comparators["(pX[" + str(node.feature) + "] <= " + str(node.threshold) + ");\n"]
				curDNFStr = _treeToDNF(node.leftChild, curDNFStr, curConjunction + "c" + str(compID) + "&&")
				curDNFStr = _treeToDNF(node.rightChild, curDNFStr, curConjunction + "!c" + str(compID) + "&&")

				return curDNFStr

		dnf = _treeToDNF(tree.head, "", "")
		code += "return " + dnf[:-2] + ";\n"
		return code

	def getCode(self, tree, treeID):
		featureType = self.getFeatureType()
		cppCode = "bool {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
					.replace("{treeID}", str(treeID)) \
					.replace("{dim}", str(self.dim)) \
					.replace("{namespace}", self.namespace) \
					.replace("{feature_t}", featureType)

		cppCode += self.getImplementation(treeID, tree)
		cppCode += "}\n"

		headerCode = "bool {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n".replace("{treeID}", str(treeID)).replace("{dim}", str(self.dim)).replace("{namespace}", self.namespace).replace("{feature_t}", featureType)

		return headerCode, cppCode

class FPGADNFTreeConverter(DNFConverter):
	def __init__(self, dim, namespace, featureType):
		super().__init__(dim, namespace, featureType)

	def getComparisonType(self):
		return "ap_uint<1>"

class X86DNFTreeConverter(DNFConverter):
	def __init__(self, dim, namespace, featureType):
		super().__init__(dim, namespace, featureType)

	def getComparisonType(self):
		return "bool"