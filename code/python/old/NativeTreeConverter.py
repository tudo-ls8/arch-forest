class FPGANativeTreeConverter(NativeTreeConverter):
        def __init__(self, dim, namespace, featureType):
                super().__init__(dim, namespace, featureType)

        def getArrayLenType(self, arrLen):
                arrayLenBit = int(np.log2(arrLen)) + 1
                return "ap_uint<"+str(arrayLenBit)+">"

        def getHeader(self, thresholdBit, treeID, arrLen):
                arrayLenBit = int(np.log2(arrLen)) + 1
                featureType = self.getFeatureType()

                headerCode = """struct {namespace}_Node{id} {
                        ap_uint<1> isLeaf;
                        ap_uint<1> prediction;
                        ap_uint<{dim}> feature;
                        ap_uint<{thresholdBit}> threshold;
                        ap_uint<{arrayLenBit}> leftChild;
                        ap_uint<{arrayLenBit}> rightChild;
                };\n""".replace("{namespace}", self.namespace) \
                           .replace("{id}", str(treeID)) \
                           .replace("{arrayLenBit}", str(arrayLenBit)) \
                           .replace("{thresholdBit}",str(thresholdBit)) \
                           .replace("{dim}", str(int(np.log2(self.dim))+1))

                headerCode += "bool {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                                .replace("{id}", str(treeID)) \
                                                .replace("{dim}", str(self.dim)) \
                                                .replace("{namespace}", self.namespace) \
                                                .replace("{feature_t}", featureType)

                return headerCode

        def getImplementation(self, head, treeID, maxDepth, avgDepth):
                arrayStructs = []
                nextIndexInArray = 1

                nodes = [head]
                while len(nodes) > 0:
                        node = nodes.pop(0)
                        entry = []

                        if node.prediction is not None:
                                entry.append(1)
                                entry.append(node.prediction)
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                        else:
                                entry.append(0)
                                entry.append(0)
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
                        bool {namespace}_predict{id}({feature_t} const pX[{dim}]){
                                {arrayLenDataType} i = 0;

                                while(!tree{id}[i].isLeaf) {
                                        #pragma HLS LOOP_TRIPCOUNT min = 0 max = {max} avg = {avg}
                                        if (pX[tree{id}[i].feature] <= tree{id}[i].threshold){
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
                   .replace("{feature_t}", featureType) \
                   .replace("{max}", str(maxDepth))\
                   .replace("{avg}", str(int(avgDepth+1)))

                return cppCode, arrLen


        def getCode(self, tree, treeID):
                maxDepth = tree.getMaxDepth()
                avgDepth = tree.getAvgDepth()

                cppCode, arrLen = self.getImplementation(tree.head, treeID, maxDepth, avgDepth)
                thresholdBit = int(np.log2(self.getMaxThreshold(tree))) + 1 if self.getMaxThreshold(tree) != 0 else 1
                headerCode = self.getHeader(thresholdBit, treeID, arrLen)

                return headerCode, cppCode