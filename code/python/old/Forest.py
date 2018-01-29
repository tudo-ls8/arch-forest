"""Summary
"""
from sklearn.tree import _tree
from functools import reduce
import json

class Forest:
        """
                Simple class to save a forest, aka multiple trees
                NOTE: This implementation is a little bit limited at the moment
                        - Currently only two class predictions (true/false) are supported.
                          Multiclass does not work!
                        - Comparisons at split nodes are performed as '<='
        """
        class Tree:
                """
                        Simple class to save a tree, aka multiple nodes
                """
                class Node:
                        """
                                Simple single node class
                        """

                        # Static variable to ensure uniques node IDs
                        nodeID = 0
                        def __init__(self):
                                # Type = Leaf node or split node (inner)
                                self.type = None

                                # The ID of this node. Makes addressing sometimes easier
                                self.id = None

                                # The total number of samples seen at this node
                                self.numSamples = None

                                # The probability of a false label at this node
                                self.negProb = None

                                # The probability of a true label at this node
                                self.posProb = None

                                # The 'way' of comparison. Usually this is "<=". However, for non-numeric
                                # features, one sometimes need equals comparisons "==". This is currently not
                                # implemented.
                                # Note: This field is only used if self.type == split
                                self.compare = None

                                # The index of the feature to compare against
                                # Note: This field is only used if self.type == split
                                self.feature = None

                                # The threashold the features gets compared against
                                # Note: This field is only used if self.type == split
                                self.threshold = None

                                # The parent index of this node in the data array
                                # Note: This is only used within the path-oriented traverse
                                self.parent = None

                                # The information that this node is the leftchild or rightchild
                                # self.side == 0 if it is the leftchild
                                # Note: This is only used within the path-oriented traverse
                                self.side = None

                                # The right child of this node inside the tree
                                # Note: This field is only used if self.type == split
                                self.rightChild = None

                                # The left child of this node inside the tree
                                # Note: This field is only used if self.type == split
                                self.leftChild = None

                                # The probability of this node accumulated from the probabilities of previous
                                # edges on the same path.
                                # Note: This field is only used after calling getProbAllPaths once
                                self.pathProb = None

                        # Unfortunately, as the standard library provides min-heap, I invert the object comparison
                        def __lt__(self, other):
                            return self.pathProb > other.pathProb
                        def __eq__(self, other):
                            return self.pathProb == other.pathProb
                        def __str__(self):
                            return str(self.id)

                        def fromSKLearn(self, tree, curNode):
                                """Generate a node from a sci-kit tree

                                Args:
                                    tree: The (internal) sci-kit tree object
                                    curNode: The index of the current node

                                Returns:
                                    Node: An node representing the given (internal) sci-kit node
                                """
                                self.numSamples = int(tree.n_node_samples[curNode])

                                if tree.n_outputs == 1:
                                        value = tree.value[curNode][0, :]
                                else:
                                        value = tree.value[curNode]

                                # NOTE: For standard Decision Trees, the value field contains the standard prediction values (absolute numbers)
                                #               For Random Forests, these values are weightes and thus need to be normalized
                                value = value / tree.weighted_n_node_samples[curNode]
                                if len(value) == 2:
                                        self.negProb = float(value[0])
                                        self.posProb = float(value[1])
                                else:
                                        print("Please implement multi class trees!")

                                self.id = Forest.Tree.Node.nodeID
                                Forest.Tree.Node.nodeID += 1

                                if tree.children_left[curNode] == _tree.TREE_LEAF and tree.children_right[curNode] == _tree.TREE_LEAF:
                                        self.type = "leaf"
                                else:
                                        self.type = "split"
                                        self.compare = "<="
                                        self.feature = tree.feature[curNode]
                                        self.threshold = int(tree.threshold[curNode])

                        def fromJSON(self, node):
                                self.id = node["id"]
                                self.type = node["type"]
                                self.numSamples = node["numSamples"]
                                self.negProb = node["negProb"]
                                self.posProb = node["posProb"]

                                if self.type == "split":
                                        self.compare = node["compare"]
                                        self.feature = node["feature"]
                                        self.threshold = node["threshold"]
                                        self.rightChild = node["rightChild"]
                                        self.leftChild = node["leftChild"]

                        def fromNode(self, node):
                                """ Simple copy constructor

                                Args:
                                    node (Node): The node to be copied

                                Returns:
                                    Node: A copy of the given node
                                """
                                self.id = node.id
                                self.type = node.type
                                self.numSamples = node.numSamples
                                self.negProb = node.negProb
                                self.posProb = node.posProb
                                self.compare = node.compare
                                self.feature = node.feature
                                self.threshold = node.threshold
                                self.rightChild = node.rightChild
                                self.leftChild = node.leftChild

                        def str(self, leftChilds = "", rightChilds = ""):
                                """ Returns a JSON-String representation of the node

                                Returns:
                                    TYPE: The JSON-String representation of the node
                                """
                                s = ""

                                s = "{"
                                s += "\"id\":" + str(self.id) + ","
                                s += "\"type\":\"" + self.type + "\","
                                s += "\"numSamples\":" + str(self.numSamples) + ","
                                s += "\"negProb\":" + str(self.negProb) + ","
                                s += "\"posProb\":" + str(self.posProb)

                                if self.type == "split":
                                        s += ",\"compare\":\"<=\","
                                        s += "\"feature\":" + str(self.feature) + ","
                                        s += "\"threshold\":" + str(self.threshold) + ","
                                        s += "\"leftChild\":" + leftChilds + ","
                                        s += "\"rightChild\": " + rightChilds
                                s += "}"

                                return s

                def __init__(self):
                        """ Generates a new tree from an internal sci-kit tree data structure

                        Args:
                            tree: A new tree object
                        """
                        # For simpler computations of statistics, we will also save all nodes in a
                        # dictionary where (key = nodeID, value = actuale node)
                        self.nodes = {}

                        # Pointer to the root node of this tree
                        self.head = None


                def fromJSON(self, tree, first = True):

                        node = self.Node()
                        node.fromJSON(tree)
                        self.nodes[node.id] = node

                        if node.type != "leaf":
                                node.rightChild = self.fromJSON(tree["rightChild"], False)
                                node.leftChild = self.fromJSON(tree["leftChild"], False)

                        if first:
                                self.head = node

                        return node

                def fromSKLearn(self, tree):
                        self.head = self._fromSKLearn(tree, 0)

                def _fromSKLearn(self, tree, curNode):
                        """ Loads a tree from sci-kit internal data structure into this object

                        Args:
                            tree (TYPE): The sci-kit tree
                            curNode (int, optional): The current node index (default = 0 ==> root node of the tree)

                        Returns:
                            TYPE: The root node of the extracted tree structure
                        """
                        node = self.Node()
                        node.fromSKLearn(tree, curNode)
                        self.nodes[node.id] = node

                        if node.type != "leaf":
                                leftChild = tree.children_left[curNode]
                                rightChild = tree.children_right[curNode]

                                node.rightChild = self._fromSKLearn(tree, rightChild)
                                node.leftChild = self._fromSKLearn(tree, leftChild)

                        return node

                def str(self, head = None):
                        if head is None:
                                head = self.head

                        if head.type == "leaf":
                                return head.str()
                        else:
                                leftChilds = self.str(head.leftChild)
                                rightChilds = self.str(head.rightChild)
                                s = head.str(leftChilds, rightChilds)
                                return s

                ## SOME STATISTICS FUNCTIONS ##

                def getNumLeaf(self):
                        return sum([1 if self.nodes[x].type == "leaf" else 0 for x in self.nodes])

                def getNumSplit(self):
                        return sum([0 if self.nodes[x].type == "leaf" else 1 for x in self.nodes])

                def getNumNodes(self):
                        return len(self.nodes)

                def getMaxDepth(self):
                        paths = self.getAllPaths()
                        return max([len(p) for p in paths])

                def getAvgDepth(self):
                        paths = self.getAllPaths()
                        return sum([len(p) for p in paths]) / len(paths)

                def getAllPaths(self, node = None, curPath = [], allPaths = []):
                        if node is None:
                                node = self.head

                        if node.type == "leaf":
                                allPaths.append(curPath)
                        else:
                                self.getAllPaths(node.leftChild, curPath + [node.negProb], allPaths)
                                self.getAllPaths(node.rightChild, curPath +[node.posProb], allPaths)

                        return allPaths

                def getMaxProb(self):
                        paths = self.getAllPaths()
                        return max( [reduce(lambda x, y: x*y, path) for path in paths] )

                def getAvgProb(self):
                        paths = self.getAllPaths()
                        return sum( [reduce(lambda x, y: x*y, path) for path in paths] ) / len(paths)

                def getProbAllPaths(self, node = None, curPath = [], allPaths = [], pathNodes = [], pathLabels = []):
                        if node is None:
                                node = self.head

                        if node.type == "leaf":
                                allPaths.append(curPath)
                                pathLabels.append(pathNodes)
                                curProb = reduce(lambda x, y: x*y, curPath)
                                node.pathProb = curProb
                                #print("Leaf nodes "+str(node.id)+" : "+str(curProb))
                        else:
                                try:
                                    pathNodes.index(node.id)
                                        #this node is root
                                except ValueError:
                                    pathNodes.append(node.id)
                                    curPath.append(1)

                                curProb = reduce(lambda x, y: x*y, curPath)
                                node.pathProb = curProb
                                #print("Root or Split nodes "+str(node.id)+ " : " +str(curProb))
                                self.getProbAllPaths(node.leftChild, curPath + [node.negProb], allPaths, pathNodes + [node.leftChild.id], pathLabels)
                                self.getProbAllPaths(node.rightChild, curPath +[node.posProb], allPaths, pathNodes + [node.rightChild.id], pathLabels)

                        return allPaths, pathLabels


        def __init__(self):
                self.trees = []

        def fromSKLearn(self,forest):
                for e in forest.estimators_:
                        tree = Forest.Tree()
                        tree.fromSKLearn(e.tree_)

                        self.trees.append(tree)

        def fromJSON(self, jsonFile):
                with open(jsonFile) as data_file:
                        data = json.load(data_file)

                for x in data:
                        tree = Forest.Tree()
                        tree.fromJSON(x)

                        self.trees.append(tree)

        def str(self):
                s = "["
                for tree in self.trees:
                        s += tree.str() + ","
                s = s[:-1] + "]"

                return s


        ## SOME STATISTICS FUNCTIONS ##

        def getAvgProb(self):
                return sum([t.getAvgProb() for t in self.trees]) / len(self.trees)

        def getMaxProb(self):
                return sum([t.getMaxProb() for t in self.trees]) / len(self.trees)

        def getAvgDepth(self):
                return sum([t.getAvgDepth() for t in self.trees]) / len(self.trees)

        def getMaxDepth(self):
                return sum([t.getMaxDepth() for t in self.trees]) / len(self.trees)

        def getAvgNumNodes(self):
                return sum([t.getNumNodes() for t in self.trees]) / len(self.trees)

        def getAvgNumPaths(self):
                return sum([t.getNumLeaf() for t in self.trees]) / len(self.trees)
