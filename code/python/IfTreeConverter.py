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

    def sizeOfSplit(self, node):
        size = 0
        if node.prediction is not None:
            raise IndexError('this node is not spilit')
        else:
            # In O0, the basic size of a split node is 4 instructions for loading.
            # Since a split node must contain a pair of if-else statements,
            # one instruction for branching is not avoidable.
            size += 5
            if node.leftChild.prediction is not None:
                size += 2
            if node.rightChild.prediction is not None:
                size += 2
            else:
                # prepare for a potential goto. This should be recalculated once gotois not necessary.
                size += 1
                # khchen:compilation should opt this with the else branch...
        #print(size)
        return size

    def getImplementation(self, treeID, head, kernel, inSize, inIdx, level = 1):
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
        labels = ""
        tabs = "".join(['\t' for i in range(level)])
        # size of i-cache is 32kB. One instruction is 32B. So there are 1024 instructions in i-cache
        budget = 100
        #print (inSize)
        #print (inIdx)
        curSize = inSize
        labelIdx = inIdx

        # khchen: swap-algorithm + kernel grouping
        if head.prediction is not None:
                if kernel is False:
                    return (code, tabs + "return " + str(int(head.prediction)) + ";\n", curSize, labelIdx)
                else:
                    return (tabs + "return " + str(int(head.prediction)) + ";\n", labels, curSize, labelIdx)
        else:
                # it is already in labels, the rest is all in labels:
                if kernel is False:
                    labels += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                    tmpOut = self.getImplementation(treeID, head.leftChild, False, curSize, labelIdx, level + 1)
                    code += tmpOut[0]
                    labels += tmpOut[1]
                    curSize = int(tmpOut[2])
                    labelIdx = int(tmpOut[3])
                    labels += tabs + "} else {\n"
                    tmpOut = self.getImplementation(treeID, head.rightChild, False, curSize, labelIdx,level + 1)
                    code += tmpOut[0]
                    labels += tmpOut[1]
                    curSize = int(tmpOut[2])
                    labelIdx = int(tmpOut[3])
                    labels += tabs + "}\n"

                else:
                    # check if it is the moment to go out the kernel
                    #print(curSize)
                    #print(self.sizeOfSplit(head))
                    if curSize + self.sizeOfSplit(head) >= budget:
                        labelIdx += 1
                        code += tabs + '\t' + "goto Label"+ str(labelIdx) + ";\n"
                        labels += "Label"+str(labelIdx)+":\n"
                        labels += "{\n"
                        if head.probLeft >= head.probRight:
                            labels += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                            tmpOut = self.getImplementation(treeID, head.leftChild, False, curSize, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            curSize = int(tmpOut[2])
                            labelIdx = int(tmpOut[3])

                            labels += tabs + "} else {\n"
                            tmpOut = self.getImplementation(treeID, head.rightChild, False, curSize, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            curSize = int(tmpOut[2])
                            labelIdx = int(tmpOut[3])
                        else:
                            labels += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                            tmpOut = self.getImplementation(treeID, head.rightChild, False, curSize, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            curSize = int(tmpOut[2])
                            labelIdx = int(tmpOut[3])
                            labels += tabs + "} else {\n"
                            tmpOut = self.getImplementation(treeID, head.leftChild, False, curSize, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            curSize = int(tmpOut[2])
                            labelIdx = int(tmpOut[3])
                        labels += tabs + "}\n"
                        labels += "}\n"
                    else:
                        curSize += self.sizeOfSplit(head)
                        if head.probLeft >= head.probRight:
                                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                                tmpOut= self.getImplementation(treeID, head.leftChild, True, curSize, labelIdx,level + 1)
                                code += tmpOut[0]
                                labels += tmpOut[1]
                                curSize = int(tmpOut[2])
                                labelIdx = int(tmpOut[3])
                                code += tabs + "} else {\n"
                                tmpOut = self.getImplementation(treeID, head.rightChild, True, curSize, labelIdx,level + 1)
                                code += tmpOut[0]
                                labels += tmpOut[1]
                                curSize = int(tmpOut[2])
                                labelIdx = int(tmpOut[3])
                        else:
                                code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                                tmpOut = self.getImplementation(treeID, head.rightChild, True, curSize, labelIdx,level + 1)
                                code += tmpOut[0]
                                labels += tmpOut[1]
                                curSize = int(tmpOut[2])
                                labelIdx = int(tmpOut[3])
                                code += tabs + "} else {\n"
                                tmpOut = self.getImplementation(treeID, head.leftChild, True, curSize, labelIdx,level + 1)
                                code += tmpOut[0]
                                labels += tmpOut[1]
                                curSize = int(tmpOut[2])
                                labelIdx = int(tmpOut[3])
                        code += tabs + "}\n"
        return (code, labels, curSize, labelIdx)

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

        #mainCode, labelsCode, curSize, labelIdx
        output = self.getImplementation(treeID, tree.head, True, 0, 0)
        cppCode += output[0]
        cppCode += output[1]
        cppCode += "}\n"

        headerCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode
