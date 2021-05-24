from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq
import gc
import os
#import objgraph

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
        # featureType = self.getFeatureType()
        # headerCode = "inline float {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]);\n" \
        #                                 .replace("{treeID}", str(treeID)) \
        #                                 .replace("{dim}", str(self.dim)) \
        #                                 .replace("{namespace}", self.namespace) \
        #                                 .replace("{feature_t}", featureType)
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            # for i in range(len(head.prediction)):
            #     code += tabs + "pred[" + str(i) + "] += " + str(head.prediction[i]) + ";\n"

            return tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n" 
            #return tabs + "return " + str(int(head.prediction)) + ";\n" ;
            #return tabs + "return " + str(float(head.prediction)) + ";\n" ;
        else:
                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                code += self.getImplementation(treeID, head.leftChild, level + 1)
                code += tabs + "} else {\n"
                code += self.getImplementation(treeID, head.rightChild, level + 1)
                code += tabs + "}\n"

        return code

    def getCode(self, tree, treeID, numClasses):
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
                                .replace("{feature_t}", featureType) \
                                .replace("{numClasses}", str(numClasses))

        cppCode += self.getImplementation(treeID, tree.head)
        cppCode += "}\n"

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType) \
                                        .replace("{numClasses}", str(numClasses))


        return headerCode, cppCode

class OptimizedIFTreeConverter(TreeConverter):

    """ A IfTreeConverter converts a DecisionTree into its if-else structure in c language
    """
    def __init__(self, dim, namespace, featureType, architecture, orientation="path", budgetSize=32*1000):
        super().__init__(dim, namespace, featureType)
        self.architecture = architecture
        if self.architecture != "arm" and self.architecture != "intel" and self.architecture != "ppc":
           raise NotImplementedError("Please use 'arm' or 'intel' or 'ppc' as target architecture - other architectures are not supported")
        self.inKernel = {}
        # size of i-cache is 32kB. One instruction is 32B. So there are 1024 instructions in i-cache
        self.givenBudget = budgetSize
        self.orientation = orientation
        if self.orientation != "path" and self.orientation != "node" and self.orientation != "swap":
            raise NotImplementedError("Please use 'path' or 'node' or 'swap' for orientation")
        self.nodeSizeTable = []
        self.treeSize = 0

    # def getPaths(self, node = None, curPath = [], allpaths = None):
    #     if node is None:
    #         node = self.head
    #     if allpaths is None:
    #         allpaths = []
    #     if node.prediction is not None:
    #         allpaths.append(curPath+[node.id])
    #     else:
    #         self.getPaths(node.leftChild, curPath + [node.id], allpaths)
    #         self.getPaths(node.rightChild, curPath + [node.id], allpaths)
    #     return allpaths

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
        #print("prepare done")
        #print("sort")
        paths = sorted(paths, key=lambda x:x[1], reverse=True)
        #print("sort done")
        #print("paths=",tmpPathes)
        #pathProbs, paths = tree.getProbAllPaths()

        # tmp = [list(x) for x in zip(*sorted(zip(pathProbs, paths), key=lambda pair: pair[0]))]
        # paths = tmp[1]
        # print("sorted=",paths)

        # paths = sorted(paths, key=lambda x: x[1])
        # allPathProb = []
        # for p in allPaths:
        #     prob = reduce(lambda x, y: x[1]*y[1], curPath)
        #     allPathProb.append((p,prob))

        # allPathProb = sorted(allPathProb,key = lambda x : x[1])
        #O(n^2) to clean up the duplicates
        # for i in paths:
        #     s.add(i)
            # if i not in s:
            #     s.append(i)
        #print(len(s))
        #test = []
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
        #print("kernel done")
        #print(tree.nodes[5].prediction)
        #print(tree.nodes[6].prediction)
        #print(test)

        '''
        sizeOfPath = []
        for path in s:
            sizeOfPath.append(reduce((lambda x, y: self.sizeOfNode(tree,tree.nodes[x])+self.sizeOfNode(tree,tree.nodes[y])), path))
            print(path)
        #print(s)

        '''

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


    def sizeOfNode(self, tree, node, featureType):
        if self.architecture == "intel":
            for i in range(len(self.nodeSizeTable)):
                if(node.id == self.nodeSizeTable[i][0]):
                    return self.nodeSizeTable[i][1]
        elif self.architecture == "arm":
            result = 0
            #split node
            if(node.prediction is None):
                result = 8
                if(featureType == 'int'):
                    if(node.feature > 31):
                        result += 2
                    if(node.feature > 1023):
                        result += 2
                    if(node.split > 255):
                        result += 2
                    if(node.rightChild.prediction is not None and node.leftChild.prediction is not None):
                        result -= 2
                elif(featureType == 'char'):
                    if(node.feature > 31):
                        result += 2
                    if(node.split > 255):
                        result += 2
                    if(node.rightChild.prediction is not None and node.leftChild.prediction is not None):
                        result -= 2
                elif(featureType == 'short'):
                    result += 2
                    if(node.split > 255):
                        result += 2
                    if(node.rightChild.prediction is not None and node.leftChild.prediction is not None):
                        result -= 2
                elif(featureType == 'float'):
                    result = 24
                    if(node.split > 255):
                        result += 4
                    if(node.rightChild.prediction is not None and node.leftChild.prediction is not None):
                        result -= 2
            #leaf node
            else:
                result = 3
                if(np.argmax(node.prediction) > 255):
                    result += 2
            self.treeSize += result
            return result

       


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
        # featureType = self.getFeatureType()
        # headerCode = "inline float {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
        #                                 .replace("{treeID}", str(treeID)) \
        #                                 .replace("{dim}", str(self.dim)) \
        #                                 .replace("{namespace}", self.namespace) \
        #                                 .replace("{feature_t}", featureType)
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        # khchen: swap-algorithm
        if head.prediction is not None:
             # for i in range(len(head.prediction)):
             #    code += tabs + "pred[" + str(i) + "] += " + str(head.prediction[i]) + ";\n"
                return tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n" 
                #return tabs + "return " + str(int(head.prediction)) + ";\n" ;
                #return tabs + "return " + str(float(head.prediction)) + ";\n" ;
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
        # featureType = self.getFeatureType()
        # headerCode = "inline float {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]);\n" \
        #                                 .replace("{treeID}", str(treeID)) \
        #                                 .replace("{dim}", str(self.dim)) \
        #                                 .replace("{namespace}", self.namespace) \
        #                                 .replace("{feature_t}", featureType) \
        #                                 .replace("{numClasses}", str(numClasses))
        code = ""
        labels = ""
        tabs = "".join(['\t' for i in range(level)])
        labelIdx = inIdx
        # khchen: swap-algorithm + kernel grouping
        if head.prediction is not None:
                # predCode = ""
                # for i in range(len(head.prediction)):
                    # predCode += tabs + "pred[" + str(i) + "] += " + str(head.prediction[i]) + ";\n"

                if self.inKernel[head.id] is False:
                    return (code, tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n", labelIdx)
                    # return (code, predCode, labelIdx)
                else:
                    return (tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n", labels,  labelIdx)
                    # return (tabs + predCode, labels,  labelIdx)
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

    def getCode(self, tree, treeID, numClasses):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
        # gc.collect()
        # objgraph.show_most_common_types(limit=20)
        # print("\tGET ALL PROBS")
        tree.getProbAllPaths()
        # print("\tDONE PROBS")  
        # gc.collect()
        # objgraph.show_most_common_types(limit=20)

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
        if self.architecture == "intel":
            self.createCppSectionFile(tree)
        '''ncFile = open("nodeCount", "a")
        ncFile.write(str(len(tree.nodes)))
        ncFile.write("\n")
        ncFile.close()'''
        if self.orientation == "path":
            self.pathSort(tree)
            output = self.getImplementation(tree, treeID, tree.head, 0)
            cppCode += output[0] #code
            cppCode += output[1] #label
        elif self.orientation == "swap":
            cppCode += self.getSwapImplementation(treeID, tree.head)
        else:
            # print("\tPATH SORT START")
            # gc.collect()
            # objgraph.show_most_common_types(limit=20)
            self.nodeSort(tree)
            output = self.getImplementation(tree, treeID, tree.head, 0)
            cppCode += output[0] #code
            cppCode += output[1] #label
            # print("\tPATH SORT DONE")
            # gc.collect()
            # objgraph.show_most_common_types(limit=20)

        #self.nodeSort(tree)
        #print("\tGET IMPL")

        #print("\tGET IMPL DONE")

        cppCode += "}\n"
        anFile = open("allNode", "a")
        anFile.write(str(treeID) + ' ' + str(self.treeSize) + '\n')
        self.treeSize = 0
        anFile.close()

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType)

        return headerCode, cppCode

    def getLeafTestCpp(self, node):
        leafTest = """__attribute__((section("test_leaf_{nid}"))) unsigned int test{nid}( {feature_t} const pX[{dim}]){
            return {value};
        }\n"""\
        .replace("{nid}", str(node.id))\
        .replace("{value}", str(int(np.argmax(node.prediction))))\
        .replace("{dim}", str(self.dim))\
        .replace("{feature_t}", self.getFeatureType())
        return leafTest
    
    def getSplitTestCpp(self, node):
        nidStr = ''
        compare = '0'
        if node.feature == 0:
            nidStr = '__'
            compare = '1'
        nidStr += str(node.id)
        splitTest = """__attribute__((section("test_split_{nid}"))) unsigned int test{nid}( {feature_t} const pX[{dim}]){
            if( pX[{cmp}] <= 20 ){
                if( pX[{feature}] <= {value} ){
                    return 10;
                }
               else { return 40; }
            }
            else{ return 30; }
        } \n"""\
        .replace("{nid}",nidStr)\
        .replace("{value}",str(node.split))\
        .replace("{dim}",str(self.dim))\
        .replace("{feature}",str(node.feature))\
        .replace("{cmp}",compare)\
        .replace("{feature_t}", self.getFeatureType())
        return splitTest
    
    def createCppSectionFile(self, tree):
        #initialize node size table
        self.nodeSizeTable = []
        #empty section define
        cppStr = """__attribute__((section("leafEmpty"))) void emp( int const pX[{dim}]){} \n"""\
            .replace("{dim}", str(self.dim))
        cppStr += """__attribute__((section("splitReturnEmpty"))) unsigned int sp_return_emp( {feature_t} const pX[{dim}]){
            return 40;
            } \n"""\
            .replace("{dim}", str(self.dim))\
            .replace("{feature_t}", self.getFeatureType())
        cppStr += """__attribute__((section("splitEmpty"))) unsigned int sp_emp( {feature_t} const pX[{dim}]){
            if( pX[0] <= 20 ){
               return 10;
            }
            else{ return 30; }
        } \n"""\
        .replace("{dim}", str(self.dim))\
        .replace("{feature_t}", self.getFeatureType())
        cppStr += """__attribute__((section("splitEmpty1"))) unsigned int sp_emp1( {feature_t} const pX[{dim}]){
            if( pX[1] <= 20 ){
               return 10;
            }
            else{ return 30; }
        } \n"""\
        .replace("{dim}", str(self.dim))\
        .replace("{feature_t}", self.getFeatureType())
        #create section file
        for i in range(len(tree.nodes)):
            if tree.nodes[i].prediction is None:
                cppStr += self.getSplitTestCpp(tree.nodes[i])
            else:
                cppStr += self.getLeafTestCpp(tree.nodes[i])
        f = open("sizeOfNode.cpp", "w")
        f.write(cppStr)
        f.close()
        os.system("g++ sizeOfNode.cpp -c -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize")
        os.system("objdump -h sizeOfNode.o > sizeOfNode")
        #read sizeOfNode
        f = open("sizeOfNode", "r")
        lineList = f.readlines()
        leafE = 0
        splitE = 0 
        splitE1 = 0
        spReturnE = 0
        #record the node size
        #sf = open("splitRecord", "w")
        #lf = open("leafRecord", "w")


        for i in range(len(lineList)):
            x = lineList[i].split()
            if(len(x) > 3):
                if(x[1][:4] == "test"):
                    if x[1][5:9] == 'leaf':
                        self.nodeSizeTable.append([ int(x[1][10:]), int( x[2], 16) - leafE])
                        #record the node size
                        #lf.write(x[1][10:] + " " + str(int( x[2], 16) - leafE) + '\n')
                    elif x[1][5:12] == 'split__':
                        self.nodeSizeTable.append([ int(x[1][13:]), int( x[2], 16) - splitE1])
                        #record the node size
                        #noid = int(x[1][13:])
                        #sf.write(str(tree.nodes[noid].feature) + ' '+ str(tree.nodes[noid].split) + " " + str(int( x[2], 16) - splitE1) + '\n')
                    else:
                        self.nodeSizeTable.append([ int(x[1][11:]), int( x[2], 16) - splitE])
                        #record the node size
                        #noid = int(x[1][11:])
                        #sf.write(str(tree.nodes[noid].feature) + ' '+ str(tree.nodes[noid].split) + " " + str(int( x[2], 16) - splitE1) + '\n')
                elif(x[1] == 'leafEmpty'):
                    leafE = int( x[2], 16)
                elif(x[1] == 'splitEmpty'):
                    splitE = int( x[2], 16) +spReturnE
                elif(x[1] == 'splitEmpty1'):
                    splitE1 = int( x[2], 16) +spReturnE
                elif(x[1] == 'splitReturnEmpty'):
                    spReturnE = int( x[2], 16) - leafE

        #sf.close()
        #lf.close()
