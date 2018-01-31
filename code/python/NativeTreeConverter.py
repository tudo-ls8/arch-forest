from ForestConverter import TreeConverter
import numpy as np
import heapq

class NativeTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

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

    def getArrayLenType(self, arrLen):
            arrayLenBit = int(np.log2(arrLen)) + 1
            if arrayLenBit <= 8:
                    arrayLenDataType = "unsigned char"
            elif arrayLenBit <= 16:
                    arrayLenDataType = "unsigned short"
            else:
                    arrayLenDataType = "unsigned int"
            return arrayLenDataType

    def getImplementation(self, head, treeID):
        raise NotImplementedError("This function should not be called directly, but only by a sub-class")

    def getHeader(self, splitType, treeID, arrLen):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            headerCode = """struct {namespace}_Node{id} {
                    bool isLeaf;
                    unsigned int prediction;
                    {dimDataType} feature;
                    {splitType} split;
                    {arrayLenDataType} leftChild;
                    {arrayLenDataType} rightChild;

            };\n""".replace("{namespace}", self.namespace) \
                       .replace("{id}", str(treeID)) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType)

            headerCode += "unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{id}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)

            return headerCode

    def getCode(self, tree, treeID):
            # kh.chen
            # Note: this function has to be called once to traverse the tree to calculate the probabilities.
            tree.getProbAllPaths()
            cppCode, arrLen = self.getImplementation(tree.head, treeID)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                if lower > 0:
                    prefix = "unsigned"
                    maxVal = upper
                else:
                    prefix = ""
                    maxVal = max(-lower, upper)

                splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if splitBit <= 8:
                    splitDataType = prefix + " char"
                elif splitBit <= 16:
                    splitDataType = prefix + " short"
                else:
                    splitDataType = prefix + " int"
            headerCode = self.getHeader(splitDataType, treeID, arrLen)

            return headerCode, cppCode

class StandardNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1

            # BFS part
            nodes = [head]
            while len(nodes) > 0:
                    node = nodes.pop(0)
                    entry = []

                    if node.prediction is not None:
                        #print("leaf:"+str(node.id))
                        entry.append(1)
                        entry.append(int(node.prediction))
                        #entry.append(node.id)
                        entry.append(0)
                        entry.append(0)
                        entry.append(0)
                        entry.append(0)
                    else:
                        entry.append(0)
                        entry.append(0) # COnstant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)
                        entry.append(nextIndexInArray)
                        nextIndexInArray += 1
                        entry.append(nextIndexInArray)
                        nextIndexInArray += 1

                        nodes.append(node.leftChild)
                        nodes.append(node.rightChild)

                    arrayStructs.append(entry)


            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)
            # kh.chen
            #print("Get ArrayLenType")
            #print(self.getArrayLenType(len(arrayStructs)))

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

            cppCode += """
                    unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]){
                            {arrayLenDataType} i = 0;

                            while(!tree{id}[i].isLeaf) {
                                    if (pX[tree{id}[i].feature] <= tree{id}[i].split){
                                            i = tree{id}[i].leftChild;
                                    } else {
                                            i = tree{id}[i].rightChild;
                                    }
                            }

                            return tree{id}[i].prediction;
                    }
            """.replace("{id}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType)

            return cppCode, arrLen

class OptimizedNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType, setSize = 3):
        super().__init__(dim, namespace, featureType)
        self.setSize = setSize

    def getImplementation(self, head, treeID):
        arrayStructs = []
        nextIndexInArray = 1

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
                #print("subroot:"+str(node.id))
                cset = []
                while len(cset) != self.setSize: # 32/10
                    #for n in L:
                        #print(n)
                    #print()
                    #print(node.id)
                    cset.append(node)
                    entry = []

                    if node.prediction is not None:
                            #print("leaf:"+str(node.id))
                            entry.append(1)
                            entry.append(int(node.prediction))
                            #entry.append(node.id)
                            entry.append(0)
                            entry.append(0)
                            entry.append(0)
                            entry.append(0)
                            arrayStructs.append(entry)

                            if node.parent != -1:
                                # if this node is not root, it must be assigned with self.side
                                if node.side == 0:
                                    arrayStructs[node.parent][4] = nextIndexInArray - 1
                                else:
                                    arrayStructs[node.parent][5] = nextIndexInArray - 1


                            nextIndexInArray += 1


                            if len(L) != 0 and len(cset) != self.setSize:
                                node = heapq.heappop(L)
                            else:
                                break
                    else:
                            #print("split:"+str(node.id))
                            entry.append(0)
                            entry.append(0) # COnstant prediction
                            #entry.append(node.id)
                            entry.append(node.feature)
                            entry.append(node.split)

                            node.leftChild.parent = nextIndexInArray - 1
                            node.rightChild.parent = nextIndexInArray - 1

                            if node.parent != -1:
                                # if this node is not root, it must be assigned with self.side
                                if node.side == 0:
                                    arrayStructs[node.parent][4] = nextIndexInArray - 1
                                else:
                                    arrayStructs[node.parent][5] = nextIndexInArray - 1

                            # the following two fields now are modified by its children.
                            entry.append(-1)
                            entry.append(-1)
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
        # kh.chen
        #print("Get ArrayLenType")
        #print(self.getArrayLenType(len(arrayStructs)))

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

        cppCode += """
                unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]){
                        {arrayLenDataType} i = 0;

                        while(!tree{id}[i].isLeaf) {
                                if (pX[tree{id}[i].feature] <= tree{id}[i].split){
                                        i = tree{id}[i].leftChild;
                                } else {
                                        i = tree{id}[i].rightChild;
                                }
                        }

                        return tree{id}[i].prediction;
                }
        """.replace("{id}", str(treeID)) \
           .replace("{dim}", str(self.dim)) \
           .replace("{namespace}", self.namespace) \
           .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
           .replace("{feature_t}", featureType)

        return cppCode, arrLen
