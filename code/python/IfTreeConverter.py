from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq

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
        headerCode = "inline unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
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
        cppCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType)

        cppCode += self.getImplementation(treeID, tree.head)
        cppCode += "}\n"

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode

class OptimizedIFTreeConverter(TreeConverter):
    """ A IfTreeConverter converts a DecisionTree into its if-else structure in c language
    """
    def __init__(self, dim, namespace, featureType, architecture, orientation="path", budgetSize=32*1000):
        super().__init__(dim, namespace, featureType)
        self.architecture = architecture
        if self.architecture != "arm" and self.architecture != "ppc" and self.architecture != "intel":
           raise NotImplementedError("Please use 'arm', 'ppc' or 'intel' as target architecture - other architectures are not supported")
        self.inKernel = {}
        # size of i-cache is 32kB. One instruction is 32B. So there are 1024 instructions in i-cache
        self.givenBudget = budgetSize
        self.orientation = orientation
        if self.orientation != "path" and self.orientation != "node" and self.orientation != "swap" and self.orientation != "subpath":
            raise NotImplementedError("Please use 'path' or 'node' or 'swap' or 'subpath' for orientation")

    # SORT ALL SUB PATH ACCORIDNG THEIR PROBABILITY
    def subPathSort(self, tree):
        self.inKernel = {}
        #print(len(self.getPaths(tree.head, [], [])))
        curSize = 0
        # s = set([])
        # flag = False
        #print("start getAllLeafPaths")
        allPath = tree.getAllPaths()
        #print("done getAllLeafPatgs")

        #print("prepare paths")
        paths = []
        for p in allPath:
            prob = 1
            path = []
            for (nid,nprob) in p:
                prob *= nprob
                path.append(nid)

            paths.append((path,prob))
        paths = sorted(paths, key=lambda x:x[1], reverse=True)
        if self.containsFloat(tree):
            splitDataType = "float"
        else:
            splitDataType = "int"

        #print("prepare kernel")
        for path in paths:
            for nodeid in path[0]:
                if not nodeid in self.inKernel:
                    if curSize >= self.givenBudget:
                        self.inKernel[nodeid] = False
                    else:
                        curSize += self.sizeOfNode(tree, tree.nodes[nodeid], splitDataType)
                        self.inKernel[nodeid] = True

    # SORT ALL PATH ACCORIDNG THEIR PROBABILITY
    def pathSort(self, tree):
        self.inKernel = {}
        #print(len(self.getPaths(tree.head, [], [])))
        curSize = 0
        # s = set([])
        # flag = False
        #print("start getAllLeafPaths")
        allPath = tree.getAllLeafPaths()
        #print("done getAllLeafPatgs")

        #print("prepare paths")
        paths = []
        for p in allPath:
            prob = 1
            path = []
            for (nid,nprob) in p:
                prob *= nprob
                path.append(nid)

            paths.append((path,prob))
        paths = sorted(paths, key=lambda x:x[1], reverse=True)
        if self.containsFloat(tree):
            splitDataType = "float"
        else:
            splitDataType = "int"

        #print("prepare kernel")
        for path in paths:
            for nodeid in path[0]:
                if not nodeid in self.inKernel:
                    if curSize >= self.givenBudget:
                        self.inKernel[nodeid] = False
                    else:
                        curSize += self.sizeOfNode(tree, tree.nodes[nodeid], splitDataType)
                        self.inKernel[nodeid] = True

    def nodeSort(self, tree):
        if self.containsFloat(tree):
            splitDataType = "float"
        else:
            splitDataType = "int"

        self.inKernel = {}
        curSize = 0
        L = []
        heapq.heapify(L)
        nodes = [tree.head]
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.leftChild is not None:
                nodes.append(node.leftChild)

            if node.rightChild is not None:
                nodes.append(node.rightChild)
            heapq.heappush(L, node)
        # now L has BFS nodes sorted by probabilities
        while len(L) > 0:
            node = heapq.heappop(L)
            curSize += self.sizeOfNode(tree,node, splitDataType)
            # if the current size is larger than budget already, break.
            if curSize >= self.givenBudget:
                self.inKernel[node.id] = False
            else:
                self.inKernel[node.id] = True


    def sizeOfNode(self, tree, node, splitDataType):
        size = 0

        if node.prediction is not None:
            if splitDataType == "int" and self.architecture == "arm":
                size += 2*4
            elif splitDataType == "float" and self.architecture == "arm":
                size += 2*4
            elif splitDataType == "int" and self.architecture == "ppc":
                size += 2*4
            elif splitDataType == "float" and self.architecture == "ppc":
                size += 2*4
            elif splitDataType == "int" and self.architecture == "intel":
                size += 10
            elif splitDataType == "float" and self.architecture == "intel":
                size += 10
        else:
            # In O0, the basic size of a split node is 4 instructions for loading.
            # Since a split node must contain a pair of if-else statements,
            # one instruction for branching is not avoidable.
            if splitDataType == "int" and self.architecture == "arm":
                # this is for arm int (ins * bytes)
                size += 5*4
            elif splitDataType == "float" and self.architecture == "arm":
                # this is for arm float
                size += 8*4
            elif splitDataType == "int" and self.architecture == "ppc":
                # this is for ppc int
                size += 5*4
            elif splitDataType == "float" and self.architecture == "ppc":
                # this is for ppc float
                size += 8*4
            elif splitDataType == "int" and self.architecture == "intel":
                # this is for intel integer (bytes)
                size += 28
            elif splitDataType == "float" and self.architecture == "intel":
                # this is for intel float (bytes)
                size += 17
        return size

    def getSwapImplementation(self, treeID, head, level = 1):
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
        headerCode = "inline unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        # khchen: swap-algorithm
        if head.prediction is not None:
                return tabs + "return " + str(int(head.prediction)) + ";\n" ;
        else:
                if head.probLeft >= head.probRight:
                        code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                        code += self.getSwapImplementation(treeID, head.leftChild, level + 1)
                        code += tabs + "} else {\n"
                        code += self.getSwapImplementation(treeID, head.rightChild, level + 1)
                        code += tabs + "}\n"
                else:
                        code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                        code += self.getSwapImplementation(treeID, head.rightChild, level + 1)
                        code += tabs + "} else {\n"
                        code += self.getSwapImplementation(treeID, head.leftChild, level + 1)
                        code += tabs + "}\n"
        return code

    def getImplementation(self, tree, treeID, head, inIdx, level = 1):
        # NOTE: USE self.setSize for INTEL / ARM sepcific set-size parameter (e.g. 3 or 6)
        # PPC requires to be checked again.
        # Node oriented.

        """ Generate the actual if-else implementation for a given node with Swapping and Kernel Grouping

        Args:
            tree : the body of this tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)
            head (TYPE): The current node to generate an if-else structure for.
            kernel (binary flag): Indicator for the case that the size of generated codes is greater than the cache.
            inIdx : Parameter for the intermediate idx of the labels
            level (int, optional): The intendation level of the generated code for easier
                                                        reading of the generated code

        Returns:
            Tuple: The string of if-else code, the string of label if-else code, generated code size and Final label index
        """
        featureType = self.getFeatureType()
        headerCode = "inline unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)
        code = ""
        labels = ""
        tabs = "".join(['\t' for i in range(level)])
        labelIdx = inIdx
        # khchen: swap-algorithm + kernel grouping
        if head.prediction is not None:
                if self.inKernel[head.id] is False:
                    return (code, tabs + "return " + str(int(head.prediction)) + ";\n", labelIdx)
                else:
                    return (tabs + "return " + str(int(head.prediction)) + ";\n", labels,  labelIdx)
        else:
                # it is split node
                # it is already in labels, the rest is all in labels:
                if self.inKernel[head.id] is False:
                    if head.probLeft >= head.probRight:
                        labels += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                        tmpOut = self.getImplementation(tree,treeID, head.leftChild, labelIdx, level + 1)
                        code += tmpOut[0]
                        labels += tmpOut[1]
                        labelIdx = int(tmpOut[2])
                        labels += tabs + "} else {\n"
                        tmpOut = self.getImplementation(tree,treeID, head.rightChild, labelIdx,level + 1)
                        code += tmpOut[0]
                        labels += tmpOut[1]
                        labelIdx = int(tmpOut[2])
                        labels += tabs + "}\n"
                    else:
                        labels += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                        tmpOut = self.getImplementation(tree,treeID, head.rightChild, labelIdx, level + 1)
                        code += tmpOut[0]
                        labels += tmpOut[1]
                        labelIdx = int(tmpOut[2])
                        labels += tabs + "} else {\n"
                        tmpOut = self.getImplementation(tree,treeID, head.leftChild, labelIdx,level + 1)
                        code += tmpOut[0]
                        labels += tmpOut[1]
                        labelIdx = int(tmpOut[2])
                        labels += tabs + "}\n"
                else:
                    # spilt is in kernel
                    if head.probLeft >= head.probRight: #swapping
                       #if the child is still in kernel
                        code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                        if self.inKernel[head.leftChild.id] is True:
                            tmpOut= self.getImplementation(tree,treeID, head.leftChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])

                        else: #if it is not, it is a moment to generate goto. The following nodes are all in labels.
                            labelIdx += 1
                            code += tabs + '\t' + "goto Label"+str(treeID)+"_"+ str(labelIdx) + ";\n"
                            labels += "Label"+str(treeID)+"_"+str(labelIdx)+":\n"
                            labels += "{\n"
                            tmpOut = self.getImplementation(tree,treeID, head.leftChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                            labels += "}\n"
                        code += tabs + "} else {\n"

                        if self.inKernel[head.rightChild.id] is True:
                            tmpOut = self.getImplementation(tree,treeID, head.rightChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])

                        else: #if it is not
                            labelIdx += 1
                            code += tabs + '\t' + "goto Label"+str(treeID)+"_"+ str(labelIdx) + ";\n"
                            labels += "Label"+str(treeID)+"_"+str(labelIdx)+":\n"
                            labels += "{\n"
                            tmpOut = self.getImplementation(tree,treeID, head.rightChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                            labels += "}\n"
                        code += tabs + "}\n"
                    else:
                       #if the child is still in kernel
                        code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                        if self.inKernel[head.rightChild.id] is True:
                            tmpOut= self.getImplementation(tree,treeID, head.rightChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                        else: #if it is not
                            labelIdx += 1
                            code += tabs + '\t' + "goto Label"+str(treeID)+"_"+ str(labelIdx) + ";\n"
                            labels += "Label"+str(treeID)+"_"+str(labelIdx)+":\n"
                            labels += "{\n"
                            tmpOut = self.getImplementation(tree,treeID, head.rightChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                            labels += "}\n"
                        code += tabs + "} else {\n"
                        if self.inKernel[head.leftChild.id] is True:
                            tmpOut = self.getImplementation(tree,treeID, head.leftChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                        else: #if it is not
                            labelIdx += 1
                            code += tabs + '\t' + "goto Label"+str(treeID)+"_"+ str(labelIdx) + ";\n"
                            labels += "Label"+str(treeID)+"_"+str(labelIdx)+":\n"
                            labels += "{\n"
                            tmpOut = self.getImplementation(tree,treeID, head.leftChild, labelIdx,level + 1)
                            code += tmpOut[0]
                            labels += tmpOut[1]
                            labelIdx = int(tmpOut[2])
                            labels += "}\n"
                        code += tabs + "}\n"
        return (code, labels, labelIdx)

    def getCode(self, tree, treeID):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
        #print("\tGET ALL PROBS")
        tree.getProbAllPaths()
        #print("\tDONE PROBS")

        featureType = self.getFeatureType()
        cppCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType)
        #print("PATH SORT")
        #self.pathSort(tree)
        #print("PATH SORT DONE")

        #self.nodeSort(tree)

        #print("GET IMPL")
        #print("\tPATH SORT")
        if self.orientation == "path":
            self.pathSort(tree)
            output = self.getImplementation(tree, treeID, tree.head, 0)
            cppCode += output[0] #code
            cppCode += output[1] #label
        elif self.orientation == "swap":
            cppCode += self.getSwapImplementation(treeID, tree.head)
        elif self.orientation == "subpath":
            self.subPathSort(tree)
            output = self.getImplementation(tree, treeID, tree.head, 0)
            cppCode += output[0] #code
            cppCode += output[1] #label
        else:
            self.nodeSort(tree)
            output = self.getImplementation(tree, treeID, tree.head, 0)
            cppCode += output[0] #code
            cppCode += output[1] #label
        #print("\tPATH SORT DONE")

        #self.nodeSort(tree)
        #print("\tGET IMPL")

        #print("\tGET IMPL DONE")

        cppCode += "}\n"

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode
