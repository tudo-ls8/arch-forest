#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import timeit

sys.path.append('../code/python')

from RandomForest import RandomForest
from ForestConverter import *
from NativeTreeConverter import *
from IfTreeConverter import *
from MixConverter import *

# A template to test the generated code
testCodeTemplate = """#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>

{headers}

void readCSV({feature_t} * XTest, unsigned int * YTest) {
	std::string line;
    std::ifstream file("{test_file}");
    unsigned int xCnt = 0;
    unsigned int yCnt = 0;

    if (file.is_open()) {
        while ( std::getline(file,line)) {
            if ( line.size() > 0) {
                std::stringstream ss(line);
                std::string entry;
                unsigned int first = true;

                while( std::getline(ss, entry,',') ) {
                    if (entry.size() > 0) {
                    	if (first) {
                    		YTest[yCnt++] = (unsigned int) atoi(entry.c_str());
                    		first = false;
                    	} else {
                    		//XTest[xCnt++] = ({feature_t}) atoi(entry.c_str());
                    		XTest[xCnt++] = ({feature_t}) atof(entry.c_str());
                    	}
                    }
                }
            }
        }
        file.close();
    }

}

int main(int argc, char const *argv[]) {

	//std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	//std :: cout << "Testing dimension:\t" << {DIM} << std :: endl;
	//std :: cout << "Feature type:\t "<< "{feature_t}" << std :: endl;
	//std :: cout << "Testing instances:\t" << {N} << std :: endl << std :: endl;
	//std :: cout << "Loading testing data..." << std :: endl;


	{allocMemory}
	readCSV(XTest,YTest);

	{measurmentCode}
	{freeMemory}

	return 1;
}
"""

measurmentCodeTemplate = """
<<<<<<< HEAD
	/* Burn-in phase to minimize cache-effect and check if data-set is okay */
	for (unsigned int i = 0; i < 2; ++i) {
=======
	for (unsigned int i = 0; i < {num_repetitions}; ++i) {
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8
		unsigned int acc = 0;
		for (unsigned int j = 0; j < {N}; ++j) {
			bool pred = {namespace}_predict(&XTest[{DIM}*j]);
			acc += (pred == YTest[j]);
		}
		if (acc != {target_acc}) {
			std :: cout << "Target accuracy was not met!" << std :: endl;
			std :: cout << "\t target: {target_acc}" << std :: endl;
			std :: cout << "\t current:" << acc << std :: endl;
<<<<<<< HEAD
			return 1;
		}
	}

	std::vector<std::chrono::milliseconds> runtimes;
	std::vector<unsigned int> accuracies;
	for (unsigned int i = 0; i < {num_repetitions}; ++i) {
		unsigned int acc = 0;
    	auto start = std::chrono::high_resolution_clock::now();
		for (unsigned int j = 0; j < {N}; ++j) {
			bool pred = {namespace}_predict(&XTest[{DIM}*j]);
			acc += (pred == YTest[j]);
		}
    	auto end = std::chrono::high_resolution_clock::now();
    	std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		runtimes.push_back((float) (duration.count() / {N}));
	}

	// Something close to welfords algorithm to estimate variance and mean on the fly
	float avg = 0.0f;
	float var = 0.0f;
	unsigned int cnt = 0;
	for (auto d : runtimes) {
		cnt++;
		float delta = d - avg;
		avg = avg + delta / cnt;
		float delta2 = d - avg;
		var = var + delta*delta2;
	}

	std :: cout << "Runtime per element (ms): " << avg << " ( " << var / (cnt - 1) << " )" <<std :: endl;
=======
		}
	}

"""

makefileTemplate = """

>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8
"""

def writeFiles(basepath, basename, header, cpp):
	if header is not None:
		with open(basepath + basename + ".h",'w') as code_file:
			code_file.write(header)

	if cpp is not None:
		with open(basepath + basename + ".cpp",'w') as code_file:
			code_file.write(cpp)

def writeTestFiles(outPath, namespace, header, dim, N, featureType, testFile, targetAcc, reps):
	allocMemory = "{feature_t} * XTest = new {feature_t}[{DIM}*{N}];\n \tunsigned int * YTest = new unsigned int[{N}];"
	freeMemory = "delete[] XTest;\n \tdelete[] YTest;"

	measurmentCode = measurmentCodeTemplate.replace("{namespace}", namespace).replace("{target_acc}", str(targetAcc)).replace("{num_repetitions}", str(reps))

	testCode = testCodeTemplate.replace("{headers}", "#include \"" + header + "\"") \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}",measurmentCode) \
							   .replace("{feature_t}", str(featureType)) \
							   .replace("{N}", str(N)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{test_file}", "../" + str(testFile)) 

	with open(outPath + "/test" + namespace + ".cpp",'w') as code_file:
		code_file.write(testCode)

def generateClassifier(outPath, X, Y, converter, namespace, featureType, forest, testFile, reps):
	YPredicted_ = forest.predict(X)
	targetAcc = sum(YPredicted_ == Y)

	# TODO: STORE NUM OF CLASSES IN TREE / FOREST ???
	# 		THIS IS ONLY NEEDED FOR CLASSIFICATION!
	numClasses = len(set(Y))

	headerCode, cppCode = converter.getCode(forest,numClasses)
	cppCode = "#include \"" + namespace + ".h\"\n" + cppCode
	writeFiles(outPath + "/", namespace, headerCode, cppCode)
	writeTestFiles(outPath, namespace, namespace + ".h", len(X[0]), len(Y), featureType, testFile, targetAcc, reps)

def getFeatureType(X):
	containsFloat = False
	for x in X:
		for xi in x:
			if isinstance(xi, float):
				containsFloat = True
				break

	if containsFloat:
		dataType = "float"
	else:
		lower = np.min(X)
		upper = np.max(X)
		if lower > 0:
			prefix = "unsigned"
			maxVal = upper
		else:
			prefix = ""
			maxVal = max(-lower, upper)

		bit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

		if bit <= 8:
			dataType = prefix + " char"
		elif bit <= 16:
			dataType = prefix + " short"
		else:
			dataType = prefix + " int"

	return dataType

def main(argv):
	if len(argv)<1:
		print("Please give a sub-folder / dataset to be used")
		return
	else:
		basepath = argv[0].strip("/")

	if len(argv)>1:
		reps = argv[1]
	else:
<<<<<<< HEAD
		reps = 10
=======
		reps = 1
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8

	for f in sorted(os.listdir(basepath + "/text/")):
		if f.endswith(".json"): 
			name = f.replace(".json","")
			dirpath = basepath + "/cpp/" + name + "/"
			if not os.path.exists(dirpath):
				os.makedirs(dirpath)
			
			testname = "/text/" + name + "_test.csv"
			#testdata = basepath + 
			forestPath = basepath + "/text/" + f

			loadedForest = RandomForest.RandomForestClassifier(None)
			loadedForest.fromJSON(forestPath) 
			#turnPoint = int(argv[0])

			data = np.genfromtxt(basepath + testname, delimiter = ",")

			X = data[:,1:]
			Y = data[:,0]

			featureType = getFeatureType(X)
			dim = len(X[0])
			numTest = len(X)

			#print("Generating If-Tree")
			converter = ForestConverter(StandardIFTreeConverter(dim, "StandardIfTree", featureType))
<<<<<<< HEAD
			generateClassifier(dirpath, X, Y, converter, "StandardIfTree", featureType, loadedForest, "../../" + testname, reps)

			converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedIfTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "OptimizedIfTree", featureType, loadedForest, "../../" + testname, reps)

			#print("Generating NativeTree")
			converter = ForestConverter(StandardNativeTreeConverter(dim, "StandardNativeTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "StandardNativeTree", featureType, loadedForest, "../../" + testname, reps)

			converter = ForestConverter(OptimizedNativeTreeConverter(dim, "OptimizedNativeTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "OptimizedNativeTree", featureType, loadedForest, "../../" + testname, reps)
=======
			generateClassifier(dirpath, X, Y, converter, "StandardIfTree", featureType, loadedForest, ".." + testname, reps)

			converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedIfTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "OptimizedIfTree", featureType, loadedForest, ".." + testname, reps)

			#print("Generating NativeTree")
			converter = ForestConverter(StandardNativeTreeConverter(dim, "StandardNativeTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "StandardNativeTree", featureType, loadedForest, ".." + testname, reps)

			converter = ForestConverter(OptimizedNativeTreeConverter(dim, "OptimizedNativeTree", featureType))
			generateClassifier(dirpath, X, Y, converter, "OptimizedNativeTree", featureType, loadedForest, ".." + testname, reps)
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8

			#print("Generating MixTree")
			#converter = MixConverter(dim, "MixTree", featureType, turnPoint)
			#generateClassifier(dirpath, X, Y, converter, "MixTree", featureType, tree, testdata)
			#$(CXX) $(FLAGS) MixTree.h MixTree.cpp testMixTree.cpp -o testMixTree

			Makefile = """CINTEL = g++
CARM = arm-linux-gnueabihf-g++ 
FLAGS = -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize

<<<<<<< HEAD
all: intel arm

arm:
=======
all: intel-cpu arm-cpu

arm-cpu:
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8
	$(CARM) $(FLAGS) StandardIfTree.h StandardIfTree.cpp testStandardIfTree.cpp -o arm/testStandardIfTree
	$(CARM) $(FLAGS) OptimizedIfTree.h OptimizedIfTree.cpp testOptimizedIfTree.cpp -o arm/testOptimizedIfTree
	$(CARM) $(FLAGS) StandardNativeTree.h StandardNativeTree.cpp testStandardNativeTree.cpp -o arm/testStandardNativeTree
	$(CARM) $(FLAGS) OptimizedNativeTree.h OptimizedNativeTree.cpp testOptimizedNativeTree.cpp -o arm/testOptimizedNativeTree

<<<<<<< HEAD
intel:
=======
intel-cpu:
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8
	$(CINTEL) $(FLAGS) StandardIfTree.h StandardIfTree.cpp testStandardIfTree.cpp -o intel/testStandardIfTree
	$(CINTEL) $(FLAGS) OptimizedIfTree.h OptimizedIfTree.cpp testOptimizedIfTree.cpp -o intel/testOptimizedIfTree
	$(CINTEL) $(FLAGS) StandardNativeTree.h StandardNativeTree.cpp testStandardNativeTree.cpp -o intel/testStandardNativeTree
	$(CINTEL) $(FLAGS) OptimizedNativeTree.h OptimizedNativeTree.cpp testOptimizedNativeTree.cpp -o intel/testOptimizedNativeTree
			"""
			with open(dirpath + "/" + "Makefile",'w') as code_file:
				code_file.write(Makefile)

if __name__ == "__main__":
   main(sys.argv[1:])
