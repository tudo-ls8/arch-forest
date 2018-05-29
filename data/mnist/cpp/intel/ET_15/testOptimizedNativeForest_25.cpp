#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <chrono>
#include <iostream>

#include "OptimizedNativeForest_25.h"

void readCSV(float * XTest, unsigned int * YTest) {
	std::string line;
    std::ifstream file("../../../test.csv");
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
                    		//XTest[xCnt++] = (float) atoi(entry.c_str());
                    		XTest[xCnt++] = (float) atof(entry.c_str());
                    	}
                    }
                }
				lineCnt++;
				if( lineCnt > 10000 ) {
					break;
				}
            }
        }
        file.close();
    }

}

int main(int argc, char const *argv[]) {

	//std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	//std :: cout << "Testing dimension:	" << 784 << std :: endl;
	//std :: cout << "Feature type:	 "<< "float" << std :: endl;
	//std :: cout << "Testing instances:	" << 10000 << std :: endl << std :: endl;
	//std :: cout << "Loading testing data..." << std :: endl;


	float * XTest = new float[784*10000];
 	unsigned int * YTest = new unsigned int[10000];
	readCSV(XTest,YTest);


	/* Burn-in phase to minimize cache-effect and check if data-set is okay */
	for (unsigned int i = 0; i < 2; ++i) {
		unsigned int acc = 0;
		for (unsigned int j = 0; j < 10000; ++j) {
			int pred = OptimizedNativeForest_25_predict(&XTest[784*j]);
			acc += (pred == YTest[j]);
			if (pred == -1)
			{
				std::cout << "\n" << "Test: A LOOP OCCURED in burn-in" << "\n";
				return 0;
			}
		}

		// SKLearn uses a weighted majority vote, whereas we use a "normal" majority vote
		// Therefore, we may not match the accuracy of SKlearn perfectly!
		if (acc != 9665) {
			std :: cout << "Target accuracy was not met!" << std :: endl;
			std :: cout << "	 target: 9665" << std :: endl;
			std :: cout << "	 current:" << acc << std :: endl;
			//return 1;
		}
	}

	std::vector<float> runtimes;
	std::vector<unsigned int> accuracies;
	// change from unsigned int to int
	int pred;
	for (unsigned int i = 0; i < 50; ++i) {
		unsigned int acc = 0;
    	auto start = std::chrono::high_resolution_clock::now();
		for (unsigned int j = 0; j < 10000; ++j) {
			pred = OptimizedNativeForest_25_predict(&XTest[784*j]);
			acc += (pred == YTest[j]);
			if (pred == -1)
			{
				std::cout << "\n" << "Test: A LOOP OCCURED" << "\n";
				return 0;
			}
			acc += pred;
		}
    	auto end = std::chrono::high_resolution_clock::now();
    	std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		runtimes.push_back((float) (duration.count() / 10000.0f));
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

	delete[] XTest;
 	delete[] YTest;

	return 1;
}
