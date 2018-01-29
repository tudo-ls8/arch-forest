#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <chrono>
#include <tuple>

#include "IfTree.h"

void readCSV(short * XTest, bool * YTest) {
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
                    		XTest[xCnt++] = (short) atoi(entry.c_str());
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
	//std :: cout << "Testing dimension:	" << 1 << std :: endl;
	//std :: cout << "Feature type:	 "<< "short" << std :: endl;
	//std :: cout << "Testing instances:	" << 4095 << std :: endl << std :: endl;
	//std :: cout << "Loading testing data..." << std :: endl;


	short * XTest = new short[1*4095];
 	bool * YTest = new bool[4095];
	readCSV(XTest,YTest);

	
	std :: vector<double> runtimes;
	std :: vector<double> accuracies;

	//std :: cout << "Time measurments for IfTree" << std :: endl;
	for (unsigned int i = 0; i < 20; ++i) {
		std::chrono::high_resolution_clock::time_point start, end;

		unsigned int acc = 0;
		start = std::chrono::high_resolution_clock::now();
		for (unsigned int j = 0; j < 50; ++j) {
			for (unsigned int i = 0; i < 4095; ++i) {
				bool pred = IfTree_predict(&XTest[1*i]);
				acc += (pred == YTest[i]);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		double runtime = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count();
		runtimes.push_back(runtime/50.0f);
		accuracies.push_back(acc/50.0f);

		//std :: cout << " Accuracy: " << (double) acc / ((double) 4095) << std :: endl;
		//std :: cout << " Throughput: " << ((double) 4095) / (double) runtime << " #elem/ms" << std :: endl;
	}

	double mean = 0;
	double var = 0;

	for (double time : runtimes) {
		double t = (double) 4095 / (double) time;
		mean += t;
	}
	mean /= runtimes.size();

	for (double time : runtimes) {
		double t = (double) 4095 / (double) time;
		var += (t - mean)*(t - mean);
	}
	var /= runtimes.size();

	//std::cout << "Throughput: " << mean << " +/- " << var << std::endl << std :: endl;

	delete[] XTest;
 	delete[] YTest;

	return 1;
}
