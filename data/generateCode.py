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
from sklearn.externals import joblib

sys.path.append('../code/python')

import sys
sys.setrecursionlimit(20000)

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
#include <chrono>

{headers}

void readCSV({feature_t} * XTest, unsigned int * YTest) {
        std::string line;
    std::ifstream file("{test_file}");
    unsigned int xCnt = 0;
    unsigned int yCnt = 0;
        unsigned int lineCnt = 0;

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
                lineCnt++;
                if( lineCnt > {N} ) {
                        break;
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
        /* Burn-in phase to minimize cache-effect and check if data-set is okay */
        for (unsigned int i = 0; i < 2; ++i) {
                unsigned int acc = 0;
                for (unsigned int j = 0; j < {N}; ++j) {
                        unsigned int pred = {namespace}_predict(&XTest[{DIM}*j]);
                        acc += (pred == YTest[j]);
                }

                // SKLearn uses a weighted majority vote, whereas we use a "normal" majority vote
                // Therefore, we may not match the accuracy of SKlearn perfectly!
                /*if (acc != {target_acc}) {
                        std :: cout << "Target accuracy was not met!" << std :: endl;
                        std :: cout << "\t target: {target_acc}" << std :: endl;
                        std :: cout << "\t current:" << acc << std :: endl;
                        //return 1;
                }*/
        }

        std::vector<float> runtimes;
        std::vector<unsigned int> accuracies;
        unsigned int pred;
        for (unsigned int i = 0; i < {num_repetitions}; ++i) {
                unsigned int acc = 0;
        auto start = std::chrono::high_resolution_clock::now();
                for (unsigned int j = 0; j < {N}; ++j) {
                        pred = {namespace}_predict(&XTest[{DIM}*j]);
                        acc += (pred == YTest[j]);
                        //acc += pred;
                }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                runtimes.push_back((float) (duration.count() / {N}.0f));
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

        //std :: cout << "Runtime per element (ms): " << avg << " ( " << var / (cnt - 1) << " )" <<std :: endl;
        std :: cout << avg << "," << var / (cnt - 1) << std :: endl;
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
                                                           .replace("{test_file}", testFile)

        with open(outPath + namespace + ".cpp",'w') as code_file:
                code_file.write(testCode)

def generateClassifier(outPath, targetAcc, DIM, N, numClasses,converter, namespace, featureType, forest, testFile, reps):
        # TODO: STORE NUM OF CLASSES IN TREE / FOREST ???
        #               THIS IS ONLY NEEDED FOR CLASSIFICATION!
        #print("GETTING THE CODE")
        headerCode, cppCode = converter.getCode(forest,numClasses)
        cppCode = "#include \"" + namespace + ".h\"\n" + cppCode
        writeFiles(outPath, namespace, headerCode, cppCode)
        writeTestFiles(outPath+"test", namespace, namespace + ".h", DIM, N, featureType, testFile, targetAcc, reps)

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
                bitUsed = 0
                if lower > 0:
                        prefix = "unsigned"
                        maxVal = upper
                else:
                        prefix = ""
                        bitUser = 1
                        maxVal = max(-lower, upper)

                bit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if bit <= (8-bitUsed):
                        dataType = prefix + " char"
                elif bit <= (16-bitUsed):
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

        if len(argv) < 2:
                print("Please give a target architecture (arm or intel)")
                return
        else:
                target = argv[1]

                if (target != "intel" and target != "arm"):
                        print("Did not recognize architecture, ", target)
                        print("Please use arm or intel")
                        return

        #if len(argv) < 3:
        if target == "intel":
                setSizes = [25]
                #budgetSizes = [128*1000, 384*1000]
                budgetSizes = [128*1024]
                #setSize = 10 # 5,10,25,50
                #budgetSize = 32*1000 # 16*1000, 32*1000, 64*1000
        else:
                #setSizes = [8,32]
                setSizes = [8]
                #budgetSizes = [32*1000, 64*1000]
                budgetSizes = [32*1024]
                        #setSize = 8 # 5,8,20,40
                        #budgetSize = 32*1000 # 16*1000, 32*1000, 64*1000
        # else:
        #       setSize = int(argv[2])
        reps = 50 # 20

        # if len(argv) < 4:
        #       reps = 20
        # else:
        #       reps = argv[2]

        if not os.path.exists(basepath + "/cpp"):
                os.makedirs(basepath + "/cpp")

        if not os.path.exists(basepath + "/cpp/" + target):
                os.makedirs(basepath + "/cpp/" + target)

        for f in sorted(os.listdir(basepath + "/text/")):
                if f.endswith(".json"):
                        name = f.replace(".json","")
                        cppPath = basepath + "/cpp/" + target + "/" + name
                        print("Generating", cppPath)

                        if not os.path.exists(cppPath):
                                os.makedirs(cppPath)

                        forestPath = basepath + "/text/" + f

                        print("\tLoading forest")
                        loadedForest = RandomForest.RandomForestClassifier(None)
                        loadedForest.fromJSON(forestPath)
                        if basepath == "synthetic-chain":
                                print(basepath+"/text/"+name+"_test.csv")
                                testname = "../../../text/" + name + "_test.csv"
                                data = np.genfromtxt(basepath + "/text/" + name + "_test.csv", delimiter = ",")
                                reps = 500
                        else:
                                # if basepath == "wearable-body-postures":
                                #       reps = 300
                                testname = "../../../test.csv"
                                data = np.genfromtxt(basepath + "/test.csv", delimiter = ",")

                        X = data[:,1:]
                        Y = data[:,0]

                        clf = joblib.load(basepath + "/text/" + name + ".pkl")
                        print("\tComputing target accuracy")
                        #YPredicted_ = loadedForest.predict(X)
                        YPredictedSK = clf.predict(X)
                        targetAcc = sum(YPredictedSK == Y)
                        #print("\tAccuracy MY:%s" % accuracy_score(Y, YPredicted_))
                        print("\ttargetAcc: %s" % sum(YPredictedSK == Y))
                        #print("\tAccuracy SK:%s" % accuracy_score(Y, YPredictedSK))
                        #print("\ttargetAcc SK: %s" % sum(YPredictedSK == Y))


                        featureType = getFeatureType(X)
                        dim = len(X[0])
                        numTest = min(len(X),10000)
                        numClasses = len(set(Y))

                        # RESET the memory
                        X = []
                        Y = []
                        Makefile = """COMPILER = {compiler}
FLAGS = -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize

all:
"""
                        # print("\tGenerating If-Trees")
                        converter = ForestConverter(StandardIFTreeConverter(dim, "StandardIfTree", featureType))
                        generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "StandardIfTree", featureType, loadedForest, testname, reps)
                        Makefile += "\t$(COMPILER) $(FLAGS) StandardIfTree.h StandardIfTree.cpp testStandardIfTree.cpp -o testStandardIfTree" + "\n"
                        for s in budgetSizes:
                                print("\tIf-Tree for budget", s)

                                converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedPathIfTree_" + str(s), featureType, target, "path", s))
                                generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "OptimizedPathIfTree_"+ str(s), featureType, loadedForest, testname, reps)
                                Makefile += "\t$(COMPILER) $(FLAGS) OptimizedPathIfTree_" + str(s)+".h" + " OptimizedPathIfTree_" + str(s)+".cpp testOptimizedPathIfTree_" + str(s)+".cpp -o testOptimizedPathIfTree_" + str(s) + "\n"

                                converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedSubPathIfTree_" + str(s), featureType, target, "subpath", s))
                                generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "OptimizedSubPathIfTree_"+ str(s), featureType, loadedForest, testname, reps)
                                Makefile += "\t$(COMPILER) $(FLAGS) OptimizedSubPathIfTree_" + str(s)+".h" + " OptimizedSubPathIfTree_" + str(s)+".cpp testOptimizedSubPathIfTree_" + str(s)+".cpp -o testOptimizedSubPathIfTree_" + str(s) + "\n"


                                # converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedNodeIfTree_" + str(s), featureType, target, "node", s))
                                # generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "OptimizedNodeIfTree_" + str(s), featureType, loadedForest, testname, reps)
                                # Makefile += "\t$(COMPILER) $(FLAGS) OptimizedNodeIfTree_" + str(s)+".h" + " OptimizedNodeIfTree_" + str(s)+".cpp testOptimizedNodeIfTree_" + str(s)+".cpp -o testOptimizedNodeIfTree_" + str(s) + "\n"

                                # converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedSwapIfTree_" + str(s), featureType, target, "swap", s))
                                # generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "OptimizedSwapIfTree_" + str(s), featureType, loadedForest, testname, reps)
                                # Makefile += "\t$(COMPILER) $(FLAGS) OptimizedSwapIfTree_" + str(s)+".h" + " OptimizedSwapIfTree_" + str(s)+".cpp testOptimizedSwapIfTree_" + str(s)+".cpp -o testOptimizedSwapIfTree_" + str(s) + "\n"

                        print("\tGenerating NativeTrees")

                        converter = ForestConverter(NaiveNativeTreeConverter(dim, "NaiveNativeTree", featureType))
                        generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "NaiveNativeTree", featureType, loadedForest, testname, reps)
                        Makefile += "\t$(COMPILER) $(FLAGS) NaiveNativeTree.h NaiveNativeTree.cpp testNaiveNativeTree.cpp -o testNaiveNativeTree\n"

                        converter = ForestConverter(StandardNativeTreeConverter(dim, "StandardNativeTree", featureType))
                        generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "StandardNativeTree", featureType, loadedForest, testname, reps)
                        Makefile += "\t$(COMPILER) $(FLAGS) StandardNativeTree.h StandardNativeTree.cpp testStandardNativeTree.cpp -o testStandardNativeTree\n"

                        for s in setSizes:
                                print("\tNative for set-size", s)

                                converter = ForestConverter(OptimizedNativeTreeConverter(dim, "OptimizedNativeTree_" + str(s), featureType, s))
                                generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "OptimizedNativeTree_" + str(s), featureType, loadedForest, testname, reps)
                                Makefile += "\t$(COMPILER) $(FLAGS) OptimizedNativeTree_" + str(s)+".h" + " OptimizedNativeTree_" + str(s)+".cpp testOptimizedNativeTree_" + str(s)+".cpp -o testOptimizedNativeTree_" + str(s) + "\n"

                        print("\tGenerating MixTrees")
                        converter = ForestConverter(MixConverter(dim, "MixTree", featureType, target))
                        generateClassifier(cppPath + "/", targetAcc, dim, numTest, numClasses, converter, "MixTree", featureType, loadedForest, testname, reps)
                        Makefile += "\t$(COMPILER) $(FLAGS) MixTree.h" + " MixTree.cpp testMixTree.cpp -o testMixTree\n"

                        if target == "intel":
                                compiler = "g++"
                        else:
                                compiler = "arm-linux-gnueabihf-g++"

                        Makefile = Makefile.replace("{compiler}", compiler)

                        with open(cppPath + "/" + "Makefile",'w') as code_file:
                                code_file.write(Makefile)
                print("")

if __name__ == "__main__":
   main(sys.argv[1:])
