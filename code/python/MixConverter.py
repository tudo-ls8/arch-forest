from ForestConverter import TreeConverter
import numpy as np
import heapq

class MixConverter(TreeConverter):
        """ A MixConverter converts a DecisionTree into its mixed structure in c language
        """
        def __init__(self, dim, namespace, featureType, architecture, orientation="path", budgetSize=32*1024):
                super().__init__(dim, namespace, featureType)
                #Generates a new mix-tree converter object
                self.architecture = architecture
                self.arrayLenBit = 0

                if self.architecture != "arm" and self.architecture != "intel":
                    raise NotImplementedError("Please use 'arm' or 'intel' as target architecture - other architectures are not supported")
                else:
                    if self.architecture == "arm":
                        self.setSize = 8
                    else:
                        self.setSize = 25
                self.inKernel = {}
                self.givenBudget = budgetSize
                self.orientation = orientation
                if self.orientation != "path" and self.orientation != "node" and self.orientation != "swap":
                    raise NotImplementedError("Please use 'path' or 'node' or 'swap' for orientation")

        def getNativeBasis(self, head, treeID):
                return self.getNativeImplementation(head, treeID)
        def pathSort(self, tree):
            self.inKernel = {}
            curSize = 0
            allPath = tree.getAllLeafPaths()

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
                elif splitDataType == "int" and self.architecture == "intel":
                    # this is for intel integer (bytes)
                    size += 28
                elif splitDataType == "float" and self.architecture == "intel":
                    # this is for intel float (bytes)
                    size += 17
            return size


        def getIFImplementation(self, tree, treeID, head, mapping, level = 1):
        #def getIFImplementation(self, tree, treeID, head, inSize, mapping, level = 1):
            # NOTE: USE self.setSize for INTEL / ARM sepcific set-size parameter (e.g. 3 or 6)

            """ Generate the actual if-else implementation for a given node with Swapping and Kernel Grouping

            Args:
                tree : the body of this tree
                treeID (TYPE): The id of this tree (in case we are dealing with a forest)
                head (TYPE): The current node to generate an if-else structure for.
                level (int, optional): The intendation level of the generated code for easier
                                                            reading of the generated code

            Returns:
                Tuple: The string of if-else code, the string of label if-else code, generated code size and Final label index
            """
            featureType = self.getFeatureType()
            headerCode = "unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            code = ""
            tabs = "".join(['\t' for i in range(level)])

            # khchen: swap-algorithm + kernel grouping
            if head.prediction is not None:
                    return (tabs + "return " + str(int(head.prediction)) + ";\n" )
            else:
                    # check if it is the moment to go out the kernel, set up the root id then goto the end of the while loop.
                    if self.inKernel[head.id] is False:
                        # set up the index before goto
                        code += tabs + '\t' + "subroot = "+str(mapping[head.id])+";\n"
                        code += tabs + '\t' + "goto Label"+str(treeID)+";\n"
                    else:
                        if head.probLeft >= head.probRight:
                                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                                code += self.getIFImplementation(tree, treeID, head.leftChild,  mapping, level + 1)
                                code += tabs + "} else {\n"
                                code += self.getIFImplementation(tree, treeID, head.rightChild,  mapping, level + 1)
                        else:
                                code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                                code += self.getIFImplementation(tree, treeID, head.rightChild,  mapping, level + 1)
                                code += tabs + "} else {\n"
                                code += self.getIFImplementation(tree, treeID, head.leftChild,  mapping, level + 1)
                        code += tabs + "}\n"
            return (code)

        def getNativeImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1

            mapping = {}

            # Path-oriented Layout
            head.parent = -1 #for root init
            L = [head]
            heapq.heapify(L)
            while len(L) > 0:
                    #print()
                    #for node in L:
                    #    print(node.pathProb)
                    #the one with the maximum probability will be the next sub-root.
                    node = heapq.heappop(L)
                    cset = []
                    while len(cset) != self.setSize: # 32/10
                        cset.append(node)
                        entry = []
                        mapping[node.id] = len(arrayStructs)
                        #if treeID == 0:
                            #print(node.id)

                        if node.prediction is not None:
                                continue
                        else:
                            entry.append(node.feature)
                            entry.append(node.split)

                            if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                                indicator = 3
                                entry.append(int(node.leftChild.prediction))
                                entry.append(int(node.rightChild.prediction))
                            elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                                indicator = 2
                                entry.append(-1)
                                node.leftChild.parent = nextIndexInArray - 1

                                entry.append(int(node.rightChild.prediction))
                            elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                                indicator = 1
                                entry.append(int(node.leftChild.prediction))
                                entry.append(-1)
                                node.rightChild.parent = nextIndexInArray - 1
                            else:
                                indicator = 0
                                entry.append(-1)
                                node.leftChild.parent = nextIndexInArray - 1
                                entry.append(-1)
                                node.rightChild.parent = nextIndexInArray - 1
                            entry.append(indicator)

                            # node.leftChild.parent = nextIndexInArray - 1
                            # node.rightChild.parent = nextIndexInArray - 1
                            if node.parent != -1:
                                # if this node is not root, it must be assigned with self.side
                                if node.side == 0:
                                    arrayStructs[node.parent][2] = nextIndexInArray - 1
                                else:
                                    arrayStructs[node.parent][3] = nextIndexInArray - 1

                            # the following two fields now are modified by its children.
                            # entry.append(-1)
                            # entry.append(-1)
                            arrayStructs.append(entry)
                            nextIndexInArray += 1

                            # note the sides of the children
                            node.leftChild.side = 0
                            node.rightChild.side = 1

                            if len(cset) != self.setSize:
                                if node.leftChild.pathProb >= node.rightChild.pathProb:
                                    heapq.heappush(L, node.rightChild)
                                    node = node.leftChild
                                else:
                                    heapq.heappush(L, node.leftChild)
                                    node = node.rightChild
                            else:
                                heapq.heappush(L, node.leftChild)
                                heapq.heappush(L, node.rightChild)

            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)

            cppCode = "{namespace}_Node{id} const tree{id}[{N}] = {" \
                    .replace("{id}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"
            #if treeID ==0:
                #print(arrayStructs)

            return cppCode, arrLen, mapping


        def getMaxThreshold(self, tree):
                return max([tree.nodes[x].split if tree.nodes[x].prediction is None else 0 for x in tree.nodes])

        def getArrayLenType(self, arrLen):
                arrayLenBit = int(np.log2(arrLen)) + 1
                if arrayLenBit <= 8:
                        arrayLenDataType = "unsigned char"
                elif arrayLenBit <= 16:
                        arrayLenDataType = "unsigned short"
                else:
                        arrayLenDataType = "unsigned int"
                return arrayLenDataType


        def getNativeHeader(self, splitType, treeID, arrLen):
                dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

                if dimBit <= 8:
                        dimDataType = "unsigned char"
                elif dimBit <= 16:
                        dimDataType = "unsigned short"
                else:
                        dimDataType = "unsigned int"

                featureType = self.getFeatureType()
                headerCode = """struct {namespace}_Node{id} {
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                        unsigned char indicator;

                };\n""".replace("{namespace}", self.namespace) \
                           .replace("{id}", str(treeID)) \
                           .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                           .replace("{splitType}",splitType) \
                           .replace("{dimDataType}",dimDataType)
                """
                headerCode += "unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                                .replace("{id}", str(treeID)) \
                                                .replace("{dim}", str(self.dim)) \
                                                .replace("{namespace}", self.namespace) \
                                                .replace("{feature_t}", featureType)
                """
                return headerCode


        def getCode(self, tree, treeID):
            """ Generate the actual mixture implementation for a given tree

            Args:
                tree (TYPE): The tree
                treeID (TYPE): The id of this tree (in case we are dealing with a forest)

            Returns:
                Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
                a *.h file and cppCode contains the code (=string) for a *.cpp file
            """
            tree.getProbAllPaths()
            featureType = self.getFeatureType()
            nativeImplementation = self.getNativeImplementation(tree.head, treeID)
            cppCode = ""
            cppCode += nativeImplementation[0]+ "\n"
            cppCode += "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                    .replace("{treeID}", str(treeID)) \
                                    .replace("{dim}", str(self.dim)) \
                                    .replace("{namespace}", self.namespace) \
                                    .replace("{feature_t}", featureType)
            cppCode += "\t unsigned int subroot;\n"
            arrLen = nativeImplementation[1]
            mapping = nativeImplementation[2]

            self.nodeSort(tree)
            ifImplementation = self.getIFImplementation(tree, treeID, tree.head, mapping, 0)
            # kernel code
            cppCode += ifImplementation

            # Data Array
            cppCode += """
Label{id}:
{
        {arrayLenDataType} i = subroot;

        while(true) {
            if (pX[tree{id}[i].feature] <= tree{id}[i].split){
                if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 2) {
                    i = tree{id}[i].leftChild;
                } else {
                    return tree{id}[i].leftChild;
                }
            } else {
                if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 1) {
                    i = tree{id}[i].rightChild;
                } else {
                    return tree{id}[i].rightChild;
                }
            }
        }
        return 0; // Make the compiler happy
}
        """.replace("{id}", str(treeID)) \
           .replace("{dim}", str(self.dim)) \
           .replace("{namespace}", self.namespace) \
           .replace("{arrayLenDataType}",self.getArrayLenType(arrLen)) \
           .replace("{feature_t}", featureType)

            cppCode += "}\n"

            # the rest is for generating the header
            headerCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                if lower > 0:
                    prefix = "unsigned"
                    maxVal = upper
                else:
                    prefix = ""
                    bitUsed = 1
                    maxVal = max(-lower, upper)

                splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if splitBit <= (8-bitUsed):
                    splitDataType = prefix + " char"
                elif splitBit <= (16-bitUsed):
                    splitDataType = prefix + " short"
                else:
                    splitDataType = prefix + " int"

            headerCode += self.getNativeHeader(splitDataType, treeID, arrLen)
            return headerCode, cppCode
