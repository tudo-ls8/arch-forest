import numpy as np 
import json 
import sys
import os
from ForestConverter import TreeConverter

class ArrayTree:
    def __init__(self,max_depth=2, min_samples_split=5, random_split=True):

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Initialize arrays 
        self.features = []
        self.threshold = []
        self.child = []

        self.delta = -1
        self.child_index = 0
        self.last_node = -1

    # parses one tree only
    def fromJSON(self, tree, node=0):
        if tree is None:
            return
        
        #if there is a leaf value, then set delta to features and the value to child
        if 'prediction' in tree: 
            self.features.append(self.delta)
            self.threshold.append(self.delta)
            if node <= self.last_node:
                node = self.last_node + 1
            self.child.extend([tree['prediction'], tree['prediction']])
            self.last_node = node + 1

        # if it is a decision node then append the info respectively
        else:
            self.features.append(tree['feature'])
            self.threshold.append(tree['split'])

            if node <= self.last_node:
                self.last_node += 1
                node = self.last_node
                left_child = node 
                right_child = left_child + 1
            else:
                left_child = node +1
                right_child = left_child + 1
            
            self.last_node = right_child

            self.child.extend([left_child, right_child])  
            self.fromJSON(tree['leftChild'], left_child)
            self.fromJSON(tree['rightChild'], right_child)

    def fromTree(self, tree, node=0, num_nodes=0):
        if tree is None:
            return
        if node == 0:
            self.last_node = -1
            self.features = []
            self.threshold = []
            self.child = np.zeros(num_nodes*2)
            self.child = self.child.tolist()

        if tree.prediction is not None:
            self.features.append(self.delta)
            self.threshold.append(self.delta)
            if node <= self.last_node:
                node = self.last_node + 1
            self.child[node] = tree.prediction
            self.child[node + 1] = tree.prediction
            self.last_node = node + 1

        else:
            self.features.append(tree.feature)
            self.threshold.append(tree.split)


            if node <= self.last_node:
                self.last_node += 1
                node = self.last_node

            self.child_index += 1
            left_child = 2 + node
            right_child = 3 + node 

            self.child[node] = self.child_index
            self.fromTree(tree.leftChild, left_child)
            self.child_index += 1
            self.child[node + 1] = self.child_index
            self.fromTree(tree.rightChild, right_child)
            
    def print_tree(self, node=0, indent="", first=True):
        ''' function to print the tree '''
        if first == True:
            self.last_node = 0

        # Check if we are at a leaf node
        if self.features[node] == self.delta:
            print(f"{indent}Predict: {self.child[node * 2]}")
            return

        # Print the decision at the current node
        print(f"{indent}X_{self.features[node]} <= {self.threshold[node]}?")

        print(f"{indent} yes ->", end="")
        self.last_node += 1
        self.print_tree(node= self.last_node, indent=indent + "  ",first=False)
        print(f"{indent} no ->", end="")
        self.last_node += 1 
        self.print_tree(node=self.last_node, indent=indent + "  ",first=False)

            

    def predict(self,X):
        print("***************************")
        print("PRINTING THE TREE")
        self.print_tree()
        print("***************************")

        preditions = [self.make_prediction(x) for x in X]
        return np.array(preditions)
    
    def make_prediction(self, x, node=0):

        # Main loop to traverse the decision tree
        while self.threshold[node] != self.delta:
            feature = self.features[node]
            threshold = self.threshold[node]

            if x[feature] <= threshold:
                node = self.child[node * 2] # go left 
            else:
                node = self.child[node * 2 + 1] # go right

            if self.features[node] == self.delta:
                break


        # Get the class associated with the leaf node
        predicted_class = self.child[node * 2] 

        return predicted_class
    
    def getCode(self, num_features, id, feature_type): # TODO: find a way for feature type
        # Generating the arrays as strings
        features_array = ', '.join(str(f) for f in self.features)
        threshold_array = ', '.join(f"{float(t):.2f}" if isinstance(t, str) else f"{t:.2f}" for t in self.threshold)
        child_array = ', '.join(f"{float(c):.1f}" for c in self.child)

        num_nodes = len(self.features)
        # Generating the header file content
        header_content = (
            f"inline unsigned int DTarr_predict{id}(float const pX[{num_features}]);\n"
        )

        # Generating the source file content
        source_content = (
            f"#include \"DTarr.h\"\n\n"
            f"int features[{num_nodes}] = {{{features_array}}};\n"
            f"float threshold[{num_nodes}]= {{{threshold_array}}};\n"
            f"float child[{num_nodes * 2}] = {{{child_array}}};\n\n"
            f"inline unsigned int DTarr_predict{id}(float const pX[{num_features}]) {{\n"
            f"    int node = 0;\n"
            f"    while (threshold[node] != -1) {{\n"
            f"        int feature = features[node];\n"
            f"        double threshold_value = threshold[node];\n"
            f"        if (pX[feature] <= threshold_value) {{\n"
            f"            node = child[node * 2];\n"
            f"        }} else {{\n"
            f"            node = child[node * 2 + 1];\n"
            f"        }}\n"
            f"        if (features[node] == -1) {{\n"
            f"            break;\n"
            f"        }}\n"
            f"    }}\n"
            f"    return child[2 * node];\n"
            f"}}\n"
        )

        return header_content, source_content

    
class ArrayTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def fromJSON(self, forest):
        with open(forest, 'r') as file:
            data = json.load(file)
        
        root = data[0]
        tree = ArrayTree()
        tree.fromJSON(root)

        tree.print_tree()   
        header, code = tree.getCode(num_samples=self.dim)
        print(header)
        print(code)

    def getCode(self, tree, treeID):
        # pass tree.head to convert to arr
        arrayTree = ArrayTree()
        arrayTree.fromTree(tree.head, num_nodes=len(tree.nodes))
        
        header, cpp = arrayTree.getCode(num_features= self.dim, id= treeID, feature_type=self.featureType)
        return header, cpp

