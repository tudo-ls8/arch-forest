#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import timeit
from sklearn.ensemble import VotingClassifier

from ForestConverter import *
from Forest import *

# A template to test the generated code
testCodeTemplate = """#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <chrono>
#include <tuple>

{featureType}
{headers}

void readCSV(feature_t * XTest, bool * YTest) {
	std::string line;
    std::ifstream file("test.txt");
    unsigned int xCnt = 0;
    unsigned int yCnt = 0;

    if (file.is_open()) {
        while ( std::getline(file,line)) {
            if ( line.size() > 0) {
                std::stringstream ss(line);
                std::string entry;
                bool first = true;

                while( std::getline(ss, entry,',') ) {
                    if (entry.size() > 0) {
                    	if (first) {
                    		YTest[yCnt++] = (bool) atoi(entry.c_str());
                    		first = false;
                    	} else {
                    		XTest[xCnt++] = (feature_t) atoi(entry.c_str());
                    	}
                    }
                }
            }
        }
        file.close();
    }

}

{testTreeCode}

int main(int argc, char const *argv[]) {

        /*
	std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	std :: cout << "Testing dimension:\t" << {DIM} << std :: endl;
	std :: cout << "Feature bit len:\t "<< {featureBitLen} << std :: endl;
	std :: cout << "Testing instances:\t" << {N} << std :: endl << std :: endl;
	std :: cout << "Loading testing data..." << std :: endl;
        */

	{allocMemory}
	readCSV(XTest,YTest);

	if (!testTrees(XTest)) {

		return 1;
	}

	{measurmentCode}
	{freeMemory}

	return 1;
}
"""
actionCodeTemplate = """
		bool p1 = IfTreeForest_predict(&XTest[{DIM}*i]);
		bool p2 = NativeTreeForest_predict(&XTest[{DIM}*i]);
		//bool p3 = DNFTreeForest_predict(&XTest[{DIM}*i]);
		bool p4 = MixTreeForest_predict(&XTest[{DIM}*i]);
"""
testTreeCodeTemplate = """
bool testTrees(feature_t const * const XTest) {
	for (unsigned int i = 0; i < {N}; ++i) {
                {typeOfTree}
/*
		bool p1 = IfTreeForest_predict(&XTest[{DIM}*i]);
		bool p2 = NativeTreeForest_predict(&XTest[{DIM}*i]);
		//bool p3 = DNFTreeForest_predict(&XTest[{DIM}*i]);
		bool p4 = MixTreeForest_predict(&XTest[{DIM}*i]);
		if (p1 != p2) {
			std :: cout << "MISMATCH ON ITEM " << i << " between NativeTrees and IfTrees" << std :: endl;
			std :: cout << "Predictions were:" << std :: endl;
			std :: cout << "IfTree: " << p1 << std :: endl;
			std :: cout << "NativeTrees: " << p2 << std :: endl;

			return false;
		}
		if (p3 != p2) {
			std :: cout << "MISMATCH ON ITEM " << i << " between NativeTrees and DNFTrees" << std :: endl;
			std :: cout << "Predictions were:" << std :: endl;
			std :: cout << "DNFTree: " << p3 << std :: endl;
			std :: cout << "NativeTrees: " << p2 << std :: endl;

			return false;
		}

		if (p4 != p2) {
			std :: cout << "MISMATCH ON ITEM " << i << " between NativeTrees and MixTrees" << std :: endl;
			std :: cout << "Predictions were:" << std :: endl;
			std :: cout << "MixTree: " << p4 << std :: endl;
			std :: cout << "NativeTrees: " << p2 << std :: endl;

			return false;
		}
*/
	}

	return true;
}
"""

measurmentCodeTemplate = """
	std :: cout << "Time measurments for {namespace}" << std :: endl;
	for (unsigned int i = 0; i < 5; ++i) {
		std::chrono::high_resolution_clock::time_point start, end;

		unsigned int acc = 0;
		start = std::chrono::high_resolution_clock::now();
		for (unsigned int i = 0; i < 100; ++i) {
			for (unsigned int j = 0; j < {N}; ++j) {
				bool pred = {namespace}_predict(&XTest[{DIM}*j]);
				acc += (pred == YTest[j]);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		double runtime = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count();

		std :: cout << " Accuracy: " << (double) acc / ((double) {N}*100) << std :: endl;
		std :: cout << " Throughput: " << ((double) {N}*100) / (double) runtime << " #elem/ms" << std :: endl;
	}
"""

makefileTemplate = """
CXX = g++
#FLAGS = -std=c++11 -Wall -O3 -funroll-loops -march=native -mtune=native -ftree-vectorize
FLAGS = -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize
HEADERS = {headers}
CPPFILES = {cppFiles}

all:
	$(CXX) $(FLAGS) $(HEADERS) $(CPPFILES) -o testTrees
perf:
	$(CXX) $(FLAGS) $(HEADERS) testMix.cpp MixTrees.cpp -o testMix
	$(CXX) $(FLAGS) $(HEADERS) testIf.cpp IfTrees.cpp -o testIf
	$(CXX) $(FLAGS) $(HEADERS) testNative.cpp NativeTrees.cpp -o testNative
	sh perf.sh
"""

perfScriptTemplate = """
perf stat -r 5 -e L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-prefetch-misses:u,L1-icache-load-misses:u,L1-icache-load:u ./testIf
perf stat -r 5 -e L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-prefetch-misses:u,L1-icache-load-misses:u,L1-icache-load:u ./testNative
perf stat -r 5 -e L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-prefetch-misses:u,L1-icache-load-misses:u,L1-icache-load:u ./testMix
"""

def writeFiles(basepath, basename, header, cpp):
	if header is not None:
		with open(basepath + basename + ".h",'w') as code_file:
			code_file.write(header)

	if cpp is not None:
		with open(basepath + basename + ".cpp",'w') as code_file:
			code_file.write(cpp)

def generateCode(forest, dim, featureBitLen, featureTypeDef, treeType, deviceType):
	if treeType == "if":
		converter = IFTreeConverter(dim)
	elif treeType == "dnf":
		converter = DNFConverter(dim, deviceType)
	elif treeType == "mix":
		converter = MixConverter(dim, deviceType)
	else:
		converter = NativeTreeConverter(dim, deviceType)

	forestConverter = ForestConverter(converter)

	cppCode = ""
	headerCode = "#ifndef {namespace}_H\n".replace("{namespace}", converter.namespace.upper())
	headerCode += "#define {namespace}_H\n".replace("{namespace}", converter.namespace.upper())
	headerCode += featureTypeDef + "\n"

	tHeaderCode, cppCode = forestConverter.getCode(forest)
	headerCode += tHeaderCode
	headerCode += "#endif"
	return headerCode, cppCode

def generateTrainingData(numFeatures, numBits, N):
	maxVal = 2**numBits - 1
	beta = np.random.normal(scale = 30, size = numFeatures)

	# beta = [-1 if i % 2 == 0 else 1 for i in range(numFeatures)]

	# # NOTE: If numFeatures is odd, we have to match the entries so that the
	# # dot product of np.dot([1 1 1 ... 1], beta) == 0
	# if numFeatures % 2 != 0:
	# 	beta[0] = 0.5
	# 	beta[1] = 0.5
	# basis = [1 for i in range(numFeatures)]

	X = []
	Y = []
	for i in range(N):
		x = np.random.randint(maxVal, size=numFeatures)
		X.append(x)
		Y.append(np.dot(beta, x) <= 0)

	X = np.array(X)
	Y = np.array(Y)
	return X,Y

def main(argv):
	if len(argv) < 1:
		print("Please give a path where the generated files should be saved")
		return

	# if len(argv) < 2:
	# 	print("Please give the type of trees to be generated")
	# 	return

	# Parameter
	outPath = argv[0]

	# treeType = argv[1]

	# The dimension of the feature vector (=length of the feature vectors)
	dim = 100

	# The size of each feature in terms of bit
	featureBitLen = 16

	# Number of training data to be generated
	NTrain = 10000

	# Number of testing data for speed (Note: This also gets saved on the disk)
	NTest = int(0.25*NTrain)


	# In case we want to read files from somewhere. For now, just use generated data

	#pathToCSVData = "../../data/fact_raw.csv"
	# print("No Binary file found on harddrive, loading from CSV")
	# csvIn = open(pathToCSVData,'r')
	# reader = csv.reader(csvIn)
	# next(reader)

	# print("Reading file")
	# X = []
	# Y = []
	# for row in reader:
	# 	if "?" not in row:
	# 	# Note: Each line ends with a comma, thus the last entry in row is empty
	# 	# Therefore, the second last entry contains the label
	# 		X.append([-1*(int(x)) for x in row[1:]])
	# 		Y.append(row[0] == "gamma")

	# X = np.array(X)
	# Y = np.array(Y)
	# print( "Shape of X " + str(X.shape) )
	# NTrain = len(X)
	# NTest = int(0.1*NTrain)

	print("Generating training data")
	X,Y = generateTrainingData(dim, featureBitLen, NTrain);

	clf = RandomForestClassifier(n_estimators=1, n_jobs=4) #number of trees
	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)
	print("Fitting model on " + str(len(XTrain)) + " data points")
	clf.fit(XTrain,YTrain)

	print("Perform prediction on " + str(len(XTest)) + " data points")
	print("Note:\tThis is not a statistical safe-and-sound cross-validation.")
	print("\tThis is just a santiy check.")

	# Compute some statistics for the fact raw data. Not usefule now
	# filteredCorrect = 0
	# filteredWrong = 0
	# cnt = 0
	# for x,y in zip(XTest, YTest):
	# 	if cnt % 10 == 0:
	# 		print("Status is : " + str(cnt))

	# 	cnt += 1
	# 	prob = clf.predict_proba(x.reshape(1, -1))[0]
	# 	if prob[0] > 0.68:
	# 		if y == 0:
	# 			filteredCorrect += 1
	# 		else:
	# 			filteredWrong += 1
		# elif prob[1] > 0.9:
		# 	if y == 1:
		# 		filteredCorrect += 1
		# 	else:
		# 		filteredWrong += 1
		# else:
		# 	pass
	# print("Filtered: " + str(filteredWrong+filteredCorrect))
	# print("Correctly filtered: " + str(filteredCorrect))

	# YPredicted = []
	# for x in XTest:
	# 	posCnt = 0
	# 	negCnt = 0
	# 	for e in clf.estimators_:
	# 		if e.predict(x.reshape(1, -1))[0]:
	# 			posCnt += 1
	# 		else:
	# 			negCnt += 1
	# 	YPredicted.append(negCnt < posCnt)

	# print(YPredicted)

	YPredicted = clf.predict(XTest)
	print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
	print("Accuracy:%s" % accuracy_score(YTest, YPredicted))

	print("Perform speed test on python native structure.")
	print("\t Note: Python already uses a simple c-backend to speed-up predictions")
	start = timeit.default_timer()
	YPredicted_ = clf.predict(X)
	end = timeit.default_timer()
	print("Throughput: " + str(float(X.shape[0]) / (float(end - start)*1000)) + " #elem/ms")

	# Generate the forest
	forest = Forest(clf)

	# Output some statistics if necessary. This may take some time :/
	# print("Avg prob: " + str(forest.getAvgProb()))
	# print("Max prob: " + str(forest.getMaxProb()))
	# print("Avg depth: " + str(forest.getAvgDepth()))
	# print("Max depth: " + str(forest.getMaxDepth()))
	# print("Avg #nodes: " + str(forest.getAvgNumNodes()))
	# print("Avg #paths: " + str(forest.getAvgNumPaths()))

	# The NativeTreeConverter also supports FPGAs code to some extend. We can ignore that
	# so far.
	treeType = "X86"
	# if treeType == "fpga":
	# 	featureType = "#include \"ap_int.h\"\n"
	# 	featureType += "typedef ap_uint<{featureBitLen}> feature_t;\n".replace("{featureBitLen}", str(featureBitLen))
	# 	allocMemory = "feature_t * XTest = (feature_t *) sds_alloc({DIM}*{N});\n \tap_uint<1> * YTest = (ap_uint<1> *) sds_alloc({N});"
	# 	freeMemory = "sds_free(XTest);\n \tsds_free(YTest);"
	# 	headers = "#include <stdlib.h>\n#include \"sds_lib.h\"\n#include \"NativeTrees.h\"\n#include \"IfTrees.h\" \n#include \"DNFTrees.h\""
	# else:
	if featureBitLen <= 8:
		featureType = "typedef unsigned char feature_t;\n"
	elif featureBitLen <= 16:
		featureType = "typedef unsigned short feature_t;\n"
	else:
		featureType = "typedef unsigned int feature_t;\n"

	allocMemory = "feature_t * XTest = new feature_t[{DIM}*{N}];\n \tbool * YTest = new bool[{N}];"
	freeMemory = "delete[] XTest;\n \tdelete[] YTest;"

	# There is also a DNFConverter, which we will not need I think
	#headers = "#include \"NativeTrees.h\"\n#include \"IfTrees.h\"\n#include \"DNFTrees.h\""
	headers = "#include \"NativeTrees.h\"\n#include \"IfTrees.h\"\n#include \"MixTrees.h\""


	print("Generating If-Trees")
	headerCode, cppCode = generateCode(forest, dim, featureBitLen, featureType, "if", treeType)
	cppCode = "#include \"IfTrees.h\"\n" + cppCode
	writeFiles(outPath + "/", "IfTrees", headerCode, cppCode)

	print("Generating native-Trees")
	headerCode, cppCode = generateCode(forest, dim, featureBitLen, featureType, "native", treeType)
	cppCode = "#include \"NativeTrees.h\"\n" + cppCode
	writeFiles(outPath + "/", "NativeTrees", headerCode, cppCode)

        # KHCHEN: Add a generation function for Mix-Trees
	print("Generating Mix-Trees")
	headerCode, cppCode = generateCode(forest, dim, featureBitLen, featureType, "mix", treeType)
	cppCode = "#include \"MixTrees.h\"\n" + cppCode
	writeFiles(outPath + "/", "MixTrees", headerCode, cppCode)
	# print("Generating dnf-Trees")
	# headerCode, cppCode = generateCode(forest, dim, featureBitLen, featureType, "dnf", treeType)
	# cppCode = "#include \"DNFTrees.h\"\n" + cppCode
	# writeFiles(outPath + "/", "DNFTrees", headerCode, cppCode)

	print("Generating test-code and makefile")

	with open(outPath + "/" + "test.txt",'w') as test_file:
		for x,y in zip(XTest, YTest):
			line = ("1" if y else "0")
			for xi in x:
				line += "," + str(xi)
			line += "\n"
			test_file.write(line)


	#measurmentCode = measurmentCodeTemplate.replace("{namespace}", "IfTreeForest") + measurmentCodeTemplate.replace("{namespace}", "NativeTreeForest") + measurmentCodeTemplate.replace("{namespace}", "DNFTreeForest")
	measurmentCode = measurmentCodeTemplate.replace("{namespace}", "IfTreeForest") + measurmentCodeTemplate.replace("{namespace}", "NativeTreeForest")+ measurmentCodeTemplate.replace("{namespace}", "MixTreeForest")

	testTreeCode=testTreeCodeTemplate.replace("{typeOfTree}", actionCodeTemplate)
	testCode = testCodeTemplate.replace("{headers}", headers) \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}",measurmentCode) \
							   .replace("{featureType}", str(featureType)) \
							   .replace("{testTreeCode}", testTreeCode) \
							   .replace("{N}", str(NTest)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{featureBitLen}", str(featureBitLen)) \

	#makefile = makefileTemplate.replace("{headers}", "NativeTrees.h IfTrees.h DNFTrees.h").replace("{cppFiles}", "testCode.cpp NativeTrees.cpp IfTrees.cpp DNFTrees.cpp")
	makefile = makefileTemplate.replace("{headers}", "NativeTrees.h IfTrees.h MixTrees.h").replace("{cppFiles}", "testCode.cpp NativeTrees.cpp IfTrees.cpp MixTrees.cpp")

	with open(outPath + "/" + "testCode.cpp",'w') as code_file:
		code_file.write(testCode)
        # KHCHEN: Prepare for perf testing
	with open(outPath + "/" + "testMix.cpp",'w') as code_file:
		actionCode = "bool p4 = MixTreeForest_predict(&XTest[{DIM}*i]);"
		testTreeCode=testTreeCodeTemplate.replace("{typeOfTree}", actionCode)
		testCode = testCodeTemplate.replace("{headers}", headers) \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
                                                           .replace("{measurmentCode}","") \
							   .replace("{featureType}", str(featureType)) \
							   .replace("{testTreeCode}", testTreeCode) \
							   .replace("{N}", str(NTest)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{featureBitLen}", str(featureBitLen))
		testCode = testCode.replace("{testTreeCode}", testTreeCode)
		code_file.write(testCode)
	with open(outPath + "/" + "testIf.cpp",'w') as code_file:
		actionCode = "bool p1 = IfTreeForest_predict(&XTest[{DIM}*i]);"
		testTreeCode=testTreeCodeTemplate.replace("{typeOfTree}", actionCode)
		testCode = testCodeTemplate.replace("{headers}", headers) \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}","") \
							   .replace("{featureType}", str(featureType)) \
							   .replace("{testTreeCode}", testTreeCode) \
							   .replace("{N}", str(NTest)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{featureBitLen}", str(featureBitLen))

		testCode = testCode.replace("{testTreeCode}", testTreeCode)
		code_file.write(testCode)
	with open(outPath + "/" + "testNative.cpp",'w') as code_file:
		actionCode = "bool p2 = NativeTreeForest_predict(&XTest[{DIM}*i]);"
		testTreeCode=testTreeCodeTemplate.replace("{typeOfTree}", actionCode)
		#print(testTreeCode)
		#print(testCode)
		testCode = testCodeTemplate.replace("{headers}", headers) \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}","") \
							   .replace("{featureType}", str(featureType)) \
							   .replace("{testTreeCode}", testTreeCode) \
							   .replace("{N}", str(NTest)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{featureBitLen}", str(featureBitLen))
		testCode = testCode.replace("{testTreeCode}", testTreeCode)
		code_file.write(testCode)

	if tree != "fpga":
		with open(outPath + "/" + "Makefile",'w') as code_file:
			code_file.write(makefile)

	with open(outPath + "/" + "perf.sh",'w') as code_file:
		code_file.write(perfScriptTemplate)
	print("Generation done.")

if __name__ == "__main__":
   main(sys.argv[1:])
