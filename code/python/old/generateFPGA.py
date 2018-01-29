#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import timeit

from Forest import *
from ForestConverter import *
from NativeTreeConverter import *
from IfTreeConverter import *
from DNFConverter import *

# A template to test the generated code
testCodeTemplate = """#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <tuple>

#include "xparameters.h"
#include "xtime_l.h"

{headers}

void readCSV({feature_t} * XTest, bool * YTest) {
	std::string line;
    std::ifstream file("test.csv");
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
                    		XTest[xCnt++] = ({feature_t}) atoi(entry.c_str());
                    	}
                    }
                }
            }
        }
        file.close();
    }

}

int main(int argc, char const *argv[]) {
    
    // BLOCK EXECUTION BEFORE START TO MAKE SURE UART CONNECTION IS PRESENT
	char input;
	std :: cin >> input;

	std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	std :: cout << "Testing dimension:\t" << {DIM} << std :: endl;
	std :: cout << "Feature type:\t "<< "{feature_t}" << std :: endl;
	std :: cout << "Testing instances:\t" << {N} << std :: endl << std :: endl;
	std :: cout << "Loading testing data..." << std :: endl;
    
	{allocMemory}
	readCSV(XTest,YTest);

	{measurmentCode}
	{freeMemory}

	return 1;
}
"""

measurmentCodeTemplate = """
	std :: vector<float> runtimes;
	std :: vector<float> accuracies;

	std :: cout << "Time measurments for {namespace}" << std :: endl;
	for (unsigned int i = 0; i < 20; ++i) {
		XTime start, end;
		
		unsigned int acc = 0;
		XTime_GetTime(&start);
		for (unsigned int j = 0; j < 5; ++j) {
			for (unsigned int i = 0; i < {N}; ++i) {
				bool pred = {namespace}_predict(&XTest[{DIM}*i]);
				acc += (pred == YTest[i]);
			}
		}
		XTime_GetTime(&end);
		float runtime =  1.0 * (end - start) / (COUNTS_PER_SECOND/1000000);
		runtimes.push_back(runtime/5.0f);
		accuracies.push_back(acc/5.0f);

		//std :: cout << " Accuracy: " << (double) acc / ((double) {N}) << std :: endl;
		//std :: cout << " Throughput: " << ((double) {N}) / (double) runtime << " #elem/ms" << std :: endl;
	}

	float mean = 0;
	float var = 0;

	for (float time : runtimes) {
		float t = (float) {N} / (float) time;
		mean += t;
	}
	mean /= runtimes.size();

	for (float time : runtimes) {
		float t = (float) {N} / (float) time;
		var += (t - mean)*(t - mean);
	}
	var /= runtimes.size();

	std::cout << "Throughput: " << mean << " +/- " << var << std::endl << std :: endl;
"""

makefileTemplate = """

"""

def writeFiles(basepath, basename, header, cpp):
	if header is not None:
		with open(basepath + basename + ".h",'w') as code_file:
			code_file.write(header)

	if cpp is not None:
		with open(basepath + basename + ".cpp",'w') as code_file:
			code_file.write(cpp)

def writeTestFiles(outPath, namespace, header, dim, N, featureType):
	allocMemory = "{feature_t} * XTest = new {feature_t}[{DIM}*{N}];\n \tbool * YTest = new bool[{N}];"
	freeMemory = "delete[] XTest;\n \tdelete[] YTest;"

	measurmentCode = measurmentCodeTemplate.replace("{namespace}", namespace) 

	testCode = testCodeTemplate.replace("{headers}", "#include \"" + header + "\"") \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}",measurmentCode) \
							   .replace("{feature_t}", str(featureType)) \
							   .replace("{N}", str(N)) \
							   .replace("{DIM}", str(dim)) \

	with open(outPath + "/test" + namespace + ".cpp",'w') as code_file:
		code_file.write(testCode)

def generateCode(forest, converter):
	forestConverter = ForestConverter(converter)

	cppCode = ""
	headerCode = "#ifndef {namespace}_H\n".replace("{namespace}", converter.namespace.upper())
	headerCode += "#define {namespace}_H\n".replace("{namespace}", converter.namespace.upper())
	headerCode += "\n"

	tHeaderCode, cppCode = forestConverter.getCode(forest)
	headerCode += tHeaderCode
	headerCode += "#endif"
	return headerCode, cppCode

def trainClassifier(XTrain, YTrain, XTest, YTest, nClassifiers):
	clf = RandomForestClassifier(n_estimators=nClassifiers, n_jobs=4) 

	print("Fitting model on " + str(len(XTrain)) + " data points")
	clf.fit(XTrain,YTrain)

	print("Perform prediction on " + str(len(XTest)) + " data points")
	print("Note:\tThis is not a statistical safe-and-sound cross-validation.")
	print("\tThis is just a santiy check.")

	YPredicted = clf.predict(XTest)
	print("Confusion matrix:\n%s" % confusion_matrix(YTest, YPredicted))
	print("Accuracy:%s" % accuracy_score(YTest, YPredicted))

	print("Perform speed test on python native structure.")
	print("\t Note: Python already uses a simple c-backend to speed-up predictions")
	
	runtimes = []
	for i in range(0, 20):
		start = timeit.default_timer()
		for j in range(0,20):
			YPredicted_ = clf.predict(XTest)

		end = timeit.default_timer()
		runtimes.append(float(XTest.shape[0]) / (float(end - start)*1000/20))

	print("Throughput: ", np.mean(runtimes), " +/-", np.var(runtimes))
		#print("Throughput: " + str(float(X.shape[0]) / (float(end - start)*1000)) + " #elem/ms")

	# Generate the forest
	forest = Forest(clf)

	return forest

def generateClassifier(outPath, XTest, classifier, converter, namespace, featureType):
	headerCode, cppCode = generateCode(classifier, converter)
	cppCode = "#include \"" + namespace + ".h\"\n" + cppCode
	writeFiles(outPath + "/", namespace, headerCode, cppCode)
	writeTestFiles(outPath, namespace, namespace + ".h", len(XTest[0]), len(XTest), featureType)

def main(argv):
	outPath = "../cpp/"

	pathToCSVData = "../../data/fact_raw.csv"
	csvIn = open(pathToCSVData,'r')
	reader = csv.reader(csvIn)
	next(reader)

	print("Reading file")
	X = []
	Y = []
	for row in reader:
		if "?" not in row:
		# Note: Each line ends with a comma, thus the last entry in row is empty
		# Therefore, the second last entry contains the label
			X.append([-1*(int(x)) for x in row[1:]])
			Y.append(row[0] == "gamma")

	X = np.array(X)
	Y = np.array(Y)
	print( "Shape of X " + str(X.shape) )
	NTrain = len(X)
	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)
	
	writer = open(outPath + "test.csv", 'w')
	for (x,y) in zip(XTest,YTest):
		s = str(1 if y else 0)
		for xi in x:
			s += "," + str(xi) 
		writer.write(s + "\n")
	writer.close()

	# Note: X does not contain negative values!
	featureBitLen = int(np.log2(X.max())) + 1
	if featureBitLen <= 8:
		featureType = "char"
	elif featureBitLen <= 16:
		featureType = "short"
	else:
		featureType = "unsigned int"
	dim = len(XTrain[0])

	print("Training classifiers")
	tree = forest = trainClassifier(XTrain, YTrain, XTest, YTest, 1)
	forest = trainClassifier(XTrain, YTrain, XTest, YTest, 50)

	print("Generating If-Tree")
	converter = IFTreeConverter(dim, "IfTree", featureType)
	generateClassifier(outPath, XTest, tree, converter, "IfTree", featureType)

	print("Generating If-Tree-Forest")
	converter = IFTreeConverter(dim, "IfTreeForest", featureType)
	generateClassifier(outPath, XTest, forest, converter, "IfTreeForest", featureType)

	print("Generating FPGANativeTree")
	converter = FPGANativeTreeConverter(dim, "FPGANativeTree", featureType)
	generateClassifier(outPath, XTest, tree, converter, "FPGANativeTree", featureType)

	print("Generating FPGANativeTree-Forest")
	converter = FPGANativeTreeConverter(dim, "FPGANativeTreeForest", featureType)
	generateClassifier(outPath, XTest, forest, converter, "FPGANativeTreeForest", featureType)

	print("Generating FPGADNF-Tree")
	converter = FPGADNFTreeConverter(dim, "FPGADNFTree", featureType)
	generateClassifier(outPath, XTest, tree, converter, "FPGADNFTree", featureType)

	print("Generating FPGADNFTree-Forest")
	converter = FPGADNFTreeConverter(dim, "FPGADNFTreeForest", featureType)
	generateClassifier(outPath, XTest, forest, converter, "FPGADNFTreeForest", featureType)

	print("Generating X86NativeTree")
	converter = X86NativeTreeConverter(dim, "X86NativeTree", featureType)
	generateClassifier(outPath, XTest, tree, converter, "X86NativeTree", featureType)

	print("Generating X86NativeTree-Forest")
	converter = X86NativeTreeConverter(dim, "X86NativeTreeForest", featureType)
	generateClassifier(outPath, XTest, forest, converter, "X86NativeTreeForest", featureType)

	print("Generating X86DNF-Tree")
	converter = X86DNFTreeConverter(dim, "X86DNFTree", featureType)
	generateClassifier(outPath, XTest, tree, converter, "X86DNFTree", featureType)

	print("Generating DNFTree-Forest")
	converter = X86DNFTreeConverter(dim, "X86DNFTreeForest", featureType)
	generateClassifier(outPath, XTest, forest, converter, "X86DNFTreeForest", featureType)

	# print("Generating dnf-Trees")
	# headerCode, cppCode = generateCode(forest, dim, featureBitLen, featureType, "dnf", treeType)
	# cppCode = "#include \"DNFTrees.h\"\n" + cppCode
	# writeFiles(outPath + "/", "DNFTrees", headerCode, cppCode)

	#measurmentCode = measurmentCodeTemplate.replace("{namespace}", "IfTreeForest") + measurmentCodeTemplate.replace("{namespace}", "NativeTreeForest") + measurmentCodeTemplate.replace("{namespace}", "DNFTreeForest")

	# + measurmentCodeTemplate.replace("{namespace}", "NativeTreeForest")

if __name__ == "__main__":
   main(sys.argv[1:])
