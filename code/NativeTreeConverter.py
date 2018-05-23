from ForestConverter import TreeConverter
import numpy as np
import heapq

class NativeTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

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

    def getHeader(self, splitType, treeID, arrLen, numClasses):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            # if (numClasses == 2):
            #     headerCode = """struct {namespace}_Node{treeID} {
            #             //bool isLeaf;
            #             //unsigned int prediction;
            #             {dimDataType} feature;
            #             {splitType} split;
            #             unsigend int leftChild; // NOTE: CAST 
            #             unsigend int rightChild;
            #             unsigned char indicator;

            #     };\n""".replace("{namespace}", self.namespace) \
            #                .replace("{treeID}", str(treeID)) \
            #                .replace("{splitType}",splitType) \
            #                .replace("{dimDataType}",dimDataType) \
            #                .replace("{numClasses}",str(numClasses))
            # else:
            headerCode = """struct {namespace}_Node{treeID} {
                    {dimDataType} feature;
                    {splitType} split;
                    //unsigned int leftChild;
                    //unsigned int rightChild;
                    float prediction[{numClasses}];
                    // IF SPLIT NODE:
                    //prediction[0] == leftChild (NOTE: Cast to unsigned int here)
                    //prediction[1] == rightChild
                    // ELSE:
                    // prediction is float array
                    unsigned char indicator;
            };\n""".replace("{namespace}", self.namespace) \
                   .replace("{treeID}", str(treeID)) \
                   .replace("{splitType}",splitType) \
                   .replace("{dimDataType}",dimDataType) \
                   .replace("{numClasses}",str(numClasses))

                        #                        //bool isLeaf;
                        # {dimDataType} feature;
                        # {splitType} split;
                        # float leftChild[{numClasses}];
                        # float rightChild[{numClasses}];
                        # unsigned char indicator;
            '''
            headerCode += "inline unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{id}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            '''
            headerCode += "inline void {namespace}_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType) \
                                            .replace("{numClasses}", str(numClasses))

            return headerCode

    def getCode(self, tree, treeID, numClasses):
            # kh.chen
            # Note: this function has to be called once to traverse the tree to calculate the probabilities.
            tree.getProbAllPaths()
            cppCode, arrLen = self.getImplementation(tree.head, treeID, numClasses)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                bitUsed = 0
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
            headerCode = self.getHeader(splitDataType, treeID, arrLen, numClasses)

            return headerCode, cppCode

class NaiveNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getHeader(self, splitType, treeID, arrLen, numClasses):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            # headerCode = """struct {namespace}_Node{id} {
            #         bool isLeaf;
            #         unsigned int prediction;
            #         {dimDataType} feature;
            #         {splitType} split;
            #         {arrayLenDataType} leftChild;
            #         {arrayLenDataType} rightChild;
            # };\n""".replace("{namespace}", self.namespace) \
            #            .replace("{id}", str(treeID)) \
            #            .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
            #            .replace("{splitType}",splitType) \
            #            .replace("{dimDataType}",dimDataType)
            headerCode = """struct {namespace}_Node{id} {
                    bool isLeaf;
                    float prediction[{numClasses}];
                    {dimDataType} feature;
                    {splitType} split;
                    {arrayLenDataType} leftChild;
                    {arrayLenDataType} rightChild;
            };\n""".replace("{namespace}", self.namespace) \
                       .replace("{id}", str(treeID)) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType) \
                       .replace("{numClasses}", str(numClasses))



            '''
            headerCode += "inline unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{id}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            '''

            headerCode += "inline void {namespace}_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType) \
                                            .replace("{numClasses}", str(numClasses))

            return headerCode

    def getImplementation(self, head, treeID, numClasses):
            arrayStructs = []
            nextIndexInArray = 1

            # BFS part
            nodes = [head]
            while len(nodes) > 0:
                node = nodes.pop(0)
                entry = []

                if node.prediction is not None:
                    entry.append(1)
                    #entry.append(int(node.prediction))
                    #entry.append(node.id)
                    entry.append(node.prediction)
                    entry.append(0)
                    entry.append(0)
                    entry.append(0)
                    entry.append(0)
                else:
                    entry.append(0)
                    #entry.append(0) # COnstant prediction
                    tmp = []
                    entry.append(tmp) # COnstant prediction
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

            cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                    .replace("{treeID}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            if type(val) is list:
                                # this is the array of predictions
                                cppCode += "{"
                                if len(val) != 0:
                                    for j in val:
                                        cppCode += str(j) + ","
                                else:
                                    for j in range(numClasses):
                                        cppCode += str(0) + ","
                                cppCode = cppCode[:-1] + "},"
                            else:
                                cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"

            # cppCode += """
            #         inline unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]){
            #                 {arrayLenDataType} i = 0;
            #                 while(!tree{id}[i].isLeaf) {
            #                         if (pX[tree{id}[i].feature] <= tree{id}[i].split){
            #                             i = tree{id}[i].leftChild;
            #                         } else {
            #                             i = tree{id}[i].rightChild;
            #                         }
            #                 }
            #                 return tree{id}[i].prediction;
            #         }
            # """.replace("{id}", str(treeID)) \
            #    .replace("{dim}", str(self.dim)) \
            #    .replace("{namespace}", self.namespace) \
            #    .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
            #    .replace("{feature_t}", featureType)

            cppCode += """
                    inline void {namespace}_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]){
                            {arrayLenDataType} i = 0;
                            while(!tree{treeID}[i].isLeaf) {
                                    if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                        i = tree{treeID}[i].leftChild;
                                    } else {
                                        i = tree{treeID}[i].rightChild;
                                    }
                            }
                            for(int j = 0; j < {numClasses}; j++)
                                pred[j] += tree{treeID}[i].prediction[j];
                    }
            """.replace("{treeID}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType) \
               .replace("{numClasses}", str(numClasses))

            return cppCode, arrLen

class StandardNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getImplementation(self, head, treeID, numClasses):
            arrayStructs = []
            nextIndexInArray = 1

            # BFS part
            nodes = [head]
            while len(nodes) > 0:
                    node = nodes.pop(0)
                    entry = []

                    if node.prediction is not None:
                        continue
                        # #print("leaf:"+str(node.id))
                        # entry.append(1)
                        # entry.append(int(node.prediction))
                        # #entry.append(node.id)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                    else:
                        #entry.append(0)
                        #entry.append(0) # COnstant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)

                        if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                            indicator = 3
                            # entry.append(int(node.leftChild.prediction))
                            # entry.append(int(node.rightChild.prediction))
                            entry.append(node.leftChild.prediction)
                            entry.append(node.rightChild.prediction)
                        elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                            indicator = 2
                            # entry.append(nextIndexInArray)
                            tmp = []
                            tmp.append(nextIndexInArray)
                            for j in range(numClasses - 1):
                                tmp.append(0)
                            entry.append(tmp)
                            nextIndexInArray += 1
                            entry.append(node.rightChild.prediction)
                            #entry.append(int(node.rightChild.prediction))
                        elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                            indicator = 1
                            entry.append(node.leftChild.prediction)
                            #entry.append(int(node.leftChild.prediction))
                            # entry.append(nextIndexInArray)
                            tmp = []
                            tmp.append(nextIndexInArray)
                            for j in range(numClasses - 1):
                                tmp.append(0)
                            entry.append(tmp)
                            nextIndexInArray += 1
                        else:
                            indicator = 0
                            # entry.append(nextIndexInArray)
                            tmp = []
                            tmp.append(nextIndexInArray)
                            for j in range(numClasses - 1):
                                tmp.append(0)
                            entry.append(tmp)
                            nextIndexInArray += 1
                            # entry.append(nextIndexInArray)
                            tmp = []
                            tmp.append(nextIndexInArray)
                            for j in range(numClasses - 1):
                                tmp.append(0)
                            entry.append(tmp)
                            nextIndexInArray += 1
                        entry.append(indicator)

                        nodes.append(node.leftChild)
                        nodes.append(node.rightChild)

                    arrayStructs.append(entry)

            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)
            # kh.chen
            #print("Get ArrayLenType")
            #print(self.getArrayLenType(len(arrayStructs)))

            cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                    .replace("{treeID}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            # for e in arrayStructs:
            #         cppCode += "{"
            #         for val in e:
            #                 cppCode += str(val) + ","
            #         cppCode = cppCode[:-1] + "},"
            # cppCode = cppCode[:-1] + "};"

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            if type(val) is list:
                                # this is the array of predictions
                                cppCode += "{"
                                if len(val) != 0:
                                    for j in val:
                                        cppCode += str(j) + ","
                                else:
                                    for j in range(numClasses):
                                        cppCode += str(0) + ","
                                cppCode = cppCode[:-1] + "},"
                            else:
                                cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"


            # cppCode += """
            #         inline unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]){
            #                 {arrayLenDataType} i = 0;

            #                 while(true) {
            #                     if (pX[tree{id}[i].feature] <= tree{id}[i].split){
            #                         if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 2) {
            #                             i = tree{id}[i].leftChild;
            #                         } else {
            #                             return tree{id}[i].leftChild;
            #                         }
            #                     } else {
            #                         if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 1) {
            #                             i = tree{id}[i].rightChild;
            #                         } else {
            #                             return tree{id}[i].rightChild;
            #                         }
            #                     }
            #                 }

            #                 return 0; // Make the compiler happy
            #         }
            # """.replace("{id}", str(treeID)) \
            #    .replace("{dim}", str(self.dim)) \
            #    .replace("{namespace}", self.namespace) \
            #    .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
            #    .replace("{feature_t}", featureType)
            cppCode += """
                    inline void {namespace}_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]){
                            {arrayLenDataType} i = 0;

                            while(true) {
                                if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 2) {
                                        i = tree{treeID}[i].leftChild[0];
                                    } else {
                                        for(int j = 0; j < {numClasses}; j++)
                                            pred[j] += tree{treeID}[i].leftChild[j];
                                        break;
                                    }
                                } else {
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 1) {
                                        i = tree{treeID}[i].rightChild[0];
                                    } else {
                                        for(int j = 0; j < {numClasses}; j++)
                                            pred[j] += tree{treeID}[i].rightChild[j];
                                        break;
                                    }
                                }
                            }
                    }
            """.replace("{treeID}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType) \
               .replace("{numClasses}", str(numClasses))
            return cppCode, arrLen

class OptimizedNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType, setSize = 3):
        super().__init__(dim, namespace, featureType)
        self.setSize = setSize

    def getImplementation(self, head, treeID, numClasses):
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
                        continue
                        #print("leaf:"+str(node.id))
                        # entry.append(1)
                        # entry.append(int(node.prediction))
                        #entry.append(node.id)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                        # arrayStructs.append(entry)

                        # if node.parent != -1:
                        #     # if this node is not root, it must be assigned with self.side
                        #     if node.side == 0:
                        #         arrayStructs[node.parent][4] = nextIndexInArray - 1
                        #     else:
                        #         arrayStructs[node.parent][5] = nextIndexInArray - 1

                        # nextIndexInArray += 1

                        # if len(L) != 0 and len(cset) != self.setSize:
                        #     node = heapq.heappop(L)
                        # else:
                        #     break
                    else:
                        #print("split:"+str(node.id))
                        #entry.append(0)
                        #entry.append(0) # Constant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)

                        tmp = []
                        if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                            indicator = 3
                            # entry.append(int(node.leftChild.prediction))
                            # entry.append(int(node.rightChild.prediction))
                            entry.append(node.leftChild.prediction)
                            entry.append(node.rightChild.prediction)
                        elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                            indicator = 2
                            entry.append(tmp)
                            node.leftChild.parent = nextIndexInArray - 1

                            # entry.append(int(node.rightChild.prediction))
                            entry.append(node.rightChild.prediction)
                        elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                            indicator = 1
                            # entry.append(int(node.leftChild.prediction))
                            entry.append(node.leftChild.prediction)
                            entry.append(tmp)
                            node.rightChild.parent = nextIndexInArray - 1

                        else:
                            indicator = 0
                            entry.append(tmp)
                            node.leftChild.parent = nextIndexInArray - 1
                            entry.append(tmp)
                            node.rightChild.parent = nextIndexInArray - 1
                        entry.append(indicator)

                        # node.leftChild.parent = nextIndexInArray - 1
                        # node.rightChild.parent = nextIndexInArray - 1
                        if node.parent != -1:
                            # if this node is not root, it must be assigned with self.side
                            if node.side == 0:
                                # arrayStructs[node.parent][2] = nextIndexInArray - 1
                                arrayStructs[node.parent][2] = [nextIndexInArray - 1]
                                for j in range(numClasses - 1):
                                    arrayStructs[node.parent][2].append(0)
                            else:
                                # arrayStructs[node.parent][3] = nextIndexInArray - 1
                                arrayStructs[node.parent][3] = [nextIndexInArray - 1]
                                for j in range(numClasses - 1):
                                    arrayStructs[node.parent][3].append(0)

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
        # kh.chen
        #print("Get ArrayLenType")
        #print(self.getArrayLenType(len(arrayStructs)))

        cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                .replace("{treeID}", str(treeID)) \
                .replace("{N}", str(len(arrayStructs))) \
                .replace("{namespace}", self.namespace)

        # for e in arrayStructs:
        #         cppCode += "{"
        #         for val in e:
        #                 cppCode += str(val) + ","
        #         cppCode = cppCode[:-1] + "},"
        # cppCode = cppCode[:-1] + "};"
        for e in arrayStructs:
                cppCode += "{"
                for val in e:
                        if type(val) is list:
                            # this is the array of predictions
                            cppCode += "{"
                            if len(val) != 0:
                                for j in val:
                                    cppCode += str(j) + ","
                            else:
                                for j in range(numClasses):
                                    cppCode += str(0) + ","
                            cppCode = cppCode[:-1] + "},"
                        else:
                            cppCode += str(val) + ","
                cppCode = cppCode[:-1] + "},"
        cppCode = cppCode[:-1] + "};"
        # cppCode += """
        #         inline unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]){
        #                     {arrayLenDataType} i = 0;

        #                     while(true) {
        #                         if (pX[tree{id}[i].feature] <= tree{id}[i].split){
        #                             if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 2) {
        #                                 i = tree{id}[i].leftChild;
        #                             } else {
        #                                 return tree{id}[i].leftChild;
        #                             }
        #                         } else {
        #                             if (tree{id}[i].indicator == 0 || tree{id}[i].indicator == 1) {
        #                                 i = tree{id}[i].rightChild;
        #                             } else {
        #                                 return tree{id}[i].rightChild;
        #                             }
        #                         }
        #                     }
        #                     return 0; // Make the compiler happy
        #             }
        # """.replace("{id}", str(treeID)) \
        #    .replace("{dim}", str(self.dim)) \
        #    .replace("{namespace}", self.namespace) \
        #    .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
        #    .replace("{feature_t}", featureType)
        cppCode += """
                inline void {namespace}_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]){
                        {arrayLenDataType} i = 0;

                        while(true) {
                            if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 2) {
                                    i = tree{treeID}[i].leftChild[0];
                                } else {
                                    for(int j = 0; j < {numClasses}; j++)
                                        pred[j] += tree{treeID}[i].leftChild[j];
                                    break;
                                }
                            } else {
                                if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 1) {
                                    i = tree{treeID}[i].rightChild[0];
                                } else {
                                    for(int j = 0; j < {numClasses}; j++)
                                        pred[j] += tree{treeID}[i].rightChild[j];
                                    break;
                                }
                            }
                        }
                }
        """.replace("{treeID}", str(treeID)) \
           .replace("{dim}", str(self.dim)) \
           .replace("{namespace}", self.namespace) \
           .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
           .replace("{feature_t}", featureType) \
           .replace("{numClasses}", str(numClasses))

        return cppCode, arrLen
