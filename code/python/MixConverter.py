import numpy as np
from ForestConverter import TreeConverter

# KHCHEN:
class MixConverter(TreeConverter):
        """ A MixConverter converts a DecisionTree into its mix structure in c language
        """
        def __init__(self, dim, namespace, featureType, turnPoint):
                super().__init__(dim, namespace, featureType)
                #Generates a new mix-tree converter object
                # TreeType
                self.treeType = namespace

                # KHCHEN: additional variable for simplicity
                self.arrayLenBit = 0
                self.nativeCode = ""
                self.nativeheaderCode = ""
                self.turnPoint = turnPoint

        def getNativeBasis(self, head, treeID):
                return self.getNativeImplementation(head, treeID, self.treeType)
                #print (self.nativeCode)

        def getIFImplementation(self, treeID, head, turnPoint, level = 1):
                """ Generate the actual if-else implementation for a given node

                Args:
                    treeID (TYPE): The id of this tree (in case we are dealing with a forest)
                    head (TYPE): The current node to generate an if-else structure for.
                    level (int, optional): The intendation level of the generated code for easier
                                                                reading of the generated code

                Returns:
                    String: The actual if-else code as a string
                """
                code = ""
                tabs = "".join(['\t' for i in range(level)])
                if head.prediction is not None:
                    predStr = "true" if head.prediction else "false"
                    return tabs + "return " + predStr + ";\n" ;
                else:
                    #KHCHEN: Only replacing split nodes is meaningful
                    if turnPoint < 1:
                        #KHCHEN: now only one array at begining
                        #self.nativeCode += self.getNativeImplementation(head, treeID, self.treeType, head.id)
                        code += tabs + "return {namespace}_predictRest{treeID}(pX, {index});\n".replace("{treeID}", str(treeID)).replace("{namespace}", self.namespace).replace("{index}",str(head.id))
                        #head.id is the idx of the considered node
                    else:
                        code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                        code += self.getIFImplementation(treeID, head.leftChild, turnPoint-1, level + 1)
                        code += tabs + "} else {\n"
                        code += self.getIFImplementation(treeID, head.rightChild, turnPoint-1, level + 1)
                        code += tabs + "}\n"
                return code

        def getNativeImplementation(self, head, treeID, treeType):
                featureType = self.getFeatureType()
                arrayStructs = []
                nextIndexInArray = 1

                nodes = [head]
                while len(nodes) > 0:
                        node = nodes.pop(0)
                        entry = []

                        if node.prediction is not None:
                                entry.append(1)
                                predStr = "true" if head.prediction else "false"
                                entry.append(predStr)
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                        else:
                                entry.append(0)
                                entry.append(0) # Constant prediction for non leaf nodes
                                entry.append(node.feature)
                                entry.append(node.split)

                                entry.append(nextIndexInArray)
                                nextIndexInArray += 1
                                entry.append(nextIndexInArray)
                                nextIndexInArray += 1

                                nodes.append(node.leftChild)
                                nodes.append(node.rightChild)
                        arrayStructs.append(entry)

                code = "{namespace}_Node{id} const tree{id}[{N}] = {".replace("{id}", str(treeID)).replace("{N}", str(len(arrayStructs))).replace("{namespace}", self.namespace)
                for e in arrayStructs:
                        code += "{"
                        for val in e:
                                code += str(val) + ","
                        code = code[:-1] + "},"
                code = code[:-1] + "};"

                arrayLenBit = int(np.log2(len(arrayStructs))) + 1
                if treeType == "fpga":
                        arrayLenDataType = "ap_uint<"+str(arrayLenBit)+">"
                else:
                        if arrayLenBit <= 8:
                                arrayLenDataType = "unsigned char"
                        elif arrayLenBit <= 16:
                                arrayLenDataType = "unsigned short"
                        else:
                                arrayLenDataType = "unsigned int"
#.replace("{nodeid}",str(nodeid))
                self.nativeheaderCode += "bool {namespace}_predictRest{id}({feature_t} const pX[{dim}], {arrayLenDataType} subroot);\n".replace("{id}", str(treeID)).replace("{dim}", str(self.dim)).replace("{namespace}", self.namespace).replace("{feature_t}",featureType).replace("{arrayLenDataType}",arrayLenDataType)
                code += """
bool {namespace}_predictRest{id}({feature_t} const pX[{dim}], {arrayLenDataType} subroot){
        {arrayLenDataType} i = subroot;

        while(!tree{id}[i].isLeaf) {
                if (pX[tree{id}[i].feature] <= tree{id}[i].threshold){
                        i = tree{id}[i].leftChild;
                } else {
                        i = tree{id}[i].rightChild;
                }
        }

        return tree{id}[i].prediction;
}
""".replace("{id}", str(treeID)).replace("{dim}", str(self.dim)).replace("{namespace}", self.namespace).replace("{arrayLenDataType}",arrayLenDataType).replace("{feature_t}",featureType)

                #KHCHEN: Write in the arrayLenBit into self variable
                if self.arrayLenBit < arrayLenBit:
                        self.arrayLenBit = arrayLenBit
                return code

        def getMaxThreshold(self, tree):
                return max([tree.nodes[x].split if tree.nodes[x].prediction is None else 0 for x in tree.nodes])

        def getArrayLenType(self, arrLen):
                raise NotImplementedError("getArrayLenType not implemented! Did you use super class?")

        def getNativeHeader(self, tree, arrayLenBit, treeID):
                thresholdBit = int(np.log2(self.getMaxThreshold(tree))) + 1 if self.getMaxThreshold(tree) != 0 else 1
                dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

                if thresholdBit <= 8:
                        thresholdDataType = "unsigned char"
                elif thresholdBit <= 16:
                        thresholdDataType = "unsigned short"
                else:
                        thresholdDataType = "unsigned int"

                if arrayLenBit <= 8:
                        arrayLenDataType = "unsigned char"
                elif arrayLenBit <= 16:
                        arrayLenDataType = "unsigned short"
                else:
                        arrayLenDataType = "unsigned int"

                if dimBit <= 8:
                        dimDataType = "unsigned char"
                elif dimBit <= 16:
                        dimDataType = "unsigned short"
                else:
                        dimDataType = "unsigned int"
                headerCode = """struct {namespace}_Node{id} {
                        bool isLeaf;
                        bool prediction;
                        {dimDataType} feature;
                        //{thresholdDataType} threshold;
                        float threshold;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                };\n""".replace("{namespace}", self.namespace).replace("{id}", str(treeID)).replace("{arrayLenDataType}", arrayLenDataType).replace("{thresholdDataType}",thresholdDataType).replace("{dimDataType}",dimDataType)

                return headerCode

        def getCode(self, tree, treeID):
                """ Generate the actual mix-tree implementation for a given tree

                Args:
                    tree (TYPE): The tree
                    treeID (TYPE): The id of this tree (in case we are dealing with a forest)

                Returns:
                    Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
                    a *.h file and cppCode contains the code (=string) for a *.cpp file
                """
                featureType = self.getFeatureType()
                headerCode = "bool {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n".replace("{treeID}", str(treeID)).replace("{dim}", str(self.dim)).replace("{namespace}", self.namespace).replace("{feature_t}", featureType)
                cppCode = "bool {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n".replace("{treeID}", str(treeID)).replace("{dim}", str(self.dim)).replace("{namespace}", self.namespace).replace("{feature_t}",featureType)
                cppCode += self.getIFImplementation(treeID, tree.head, self.turnPoint)
                #print("Max:"+str(tree.getMaxDepth()))
                #print("Avg:"+str(tree.getAvgDepth()))
                cppCode += "}\n"
                cppCode += self.getNativeBasis(tree.head, treeID) + "\n"
                #cppCode += self.nativeCode + "\n"
                #KHCHEN: Glue the native function prototypes into the header.
                headerCode += self.nativeheaderCode + "\n"
                #KHCHEN: Glue in the structure
                headerCode += self.getNativeHeader(tree, self.arrayLenBit, treeID)
                self.nativeCode = ""
                self.nativeheaderCode = ""
                return headerCode, cppCode
