#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <chrono>
#include <tuple>

#include "NativeTree.h"

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

    short * XTest = new short[1*2047];
 	bool * YTest = new bool[2047];
	readCSV(XTest,YTest);
    // instance 1
    for (unsigned int i = 0; i<1000000; ++i)
        NativeTree_predict(&XTest[1]);
}
/*
int main(int argc, char const *argv[]) {
    
    std :: cout << sizeof(NativeTree_Node0) << std :: endl;
    std :: cout << sizeof(bool) << std :: endl;
    std :: cout << sizeof(unsigned char) << std :: endl;
    std :: cout << sizeof(unsigned short) << std :: endl;

	//std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	//std :: cout << "Testing dimension:	" << 1 << std :: endl;
	//std :: cout << "Feature type:	 "<< "short" << std :: endl;
	//std :: cout << "Testing instances:	" << 2047 << std :: endl << std :: endl;
	//std :: cout << "Loading testing data..." << std :: endl;


	short * XTest = new short[1*2047];
 	bool * YTest = new bool[2047];
	readCSV(XTest,YTest);

	
	std :: vector<double> runtimes;
	std :: vector<double> accuracies;

	//std :: cout << "Time measurments for NativeTree" << std :: endl;
	for (unsigned int i = 0; i < 20; ++i) {
		std::chrono::high_resolution_clock::time_point start, end;

		unsigned int acc = 0;
		start = std::chrono::high_resolution_clock::now();
		for (unsigned int j = 0; j < 50; ++j) {
			for (unsigned int i = 0; i < 2047; ++i) {
				bool pred = NativeTree_predict(&XTest[1*i]);
				acc += (pred == YTest[i]);
			}
		}
		end = std::chrono::high_resolution_clock::now();
		double runtime = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count();
		runtimes.push_back(runtime/50.0f);
		accuracies.push_back(acc/50.0f);

		//std :: cout << " Accuracy: " << (double) acc / ((double) 2047) << std :: endl;
		//std :: cout << " Throughput: " << ((double) 2047) / (double) runtime << " #elem/ms" << std :: endl;
	}

	double mean = 0;
	double var = 0;

	for (double time : runtimes) {
		double t = (double) 2047 / (double) time;
		mean += t;
	}
	mean /= runtimes.size();

	for (double time : runtimes) {
		double t = (double) 2047 / (double) time;
		var += (t - mean)*(t - mean);
	}
	var /= runtimes.size();

	//std::cout << "Throughput: " << mean << " +/- " << var << std::endl << std :: endl;

	delete[] XTest;
 	delete[] YTest;

	return 1;
}
*/
