from ForestConverter import TreeConverter

class StandardIFTreeConverter(TreeConverter):
    """ A IfTreeConverter converts a DecisionTree into its if-else structure in c language
    """
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1):
        """ Generate the actual if-else implementation for a given node

        Args:
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)
            head (TYPE): The current node to generate an if-else structure for.
            level (int, optional): The intendation level of the generated code for easier
                                                        reading of the generated code

        Returns:
            String: The actual if-else code as a string
        """
        featureType = self.getFeatureType()
        headerCode = "unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            return tabs + "return " + str(int(head.prediction)) + ";\n" ;
        else:
                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                code += self.getImplementation(treeID, head.leftChild, level + 1)
                code += tabs + "} else {\n"
                code += self.getImplementation(treeID, head.rightChild, level + 1)
                code += tabs + "}\n"

        return code

    def getCode(self, tree, treeID):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
        featureType = self.getFeatureType()
        cppCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType)

        cppCode += self.getImplementation(treeID, tree.head)
        cppCode += "}\n"

        headerCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode

class OptimizedIFTreeConverter(TreeConverter):
    """ A IfTreeConverter converts a DecisionTree into its if-else structure in c language
    """
    def __init__(self, dim, namespace, featureType, setSize):
        super().__init__(dim, namespace, featureType)
        self.setSize = setSize

    def getImplementation(self, treeID, head, kernel, level = 1):
        # NOTE: USE self.setSize for INTEL / ARM sepcific set-size parameter (e.g. 3 or 6)

        """ Generate the actual if-else implementation for a given node

        Args:
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)
            head (TYPE): The current node to generate an if-else structure for.
            level (int, optional): The intendation level of the generated code for easier
                                                        reading of the generated code

        Returns:
            String: The actual if-else code as a string
        """
        featureType = self.getFeatureType()
        headerCode = "unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)
        code = ""
        labels = "{\n"
        tabs = "".join(['\t' for i in range(level)])

        # khchen: swap-algorithm
        if head.prediction is not None:
                return tabs + "return " + str(int(head.prediction)) + ";\n"
        else:
                # it is already in labels, the rest is all in labels:
                if kernel is False:
                    labels += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                    labels += self.getImplementation(treeID, head.leftChild, False, level + 1)
                    labels += tabs + "} else {\n"
                    labels += self.getImplementation(treeID, head.rightChild, False, level + 1)
                    labels += tabs + "}\n"
                else:
                    # check if it is the moment to go out the kernel
                    if curSize + function(head) >= budget:
                        code += tabs + '\t' + "goto Label"+ str(which label) + ";\n"
                        labels += "Label"+str(which label)+":\n"
                        if head.probLeft >= head.probRight:
                            labels += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                            code, labels += self.getImplementation(treeID, head.leftChild, False, level + 1)
                            labels += tabs + "} else {\n"
                            code, labels += self.getImplementation(treeID, head.rightChild, False, level + 1)
                        else:
                            labels += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                            code, labels += self.getImplementation(treeID, head.rightChild, False, level + 1)
                            labels += tabs + "} else {\n"
                            code, labels += self.getImplementation(treeID, head.leftChild, False, level + 1)
                        labels += tabs + "}\n"
                        labels += "}\n"
                    else:
                        if head.probLeft >= head.probRight:
                                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                                code, labels += self.getImplementation(treeID, head.leftChild, True, level + 1)
                                code += tabs + "} else {\n"
                                code, labels += self.getImplementation(treeID, head.rightChild, True, level + 1)
                        else:
                                code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                                code, labels += self.getImplementation(treeID, head.rightChild, True, level + 1)
                                code += tabs + "} else {\n"
                                code, labels += self.getImplementation(treeID, head.leftChild, True, level + 1)
                        code += tabs + "}\n"
        return code, labels

    def getCode(self, tree, treeID):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
        featureType = self.getFeatureType()
        cppCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType)

        mainCode, labelsCode = self.getImplementation(treeID, tree.head, True)
        cppCode += mainCode
        cppCode += labelsCode
        cppCode += "}\n"

        headerCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode
