CXX = g++
FLAGS = -std=c++11 -Wall -O0 -funroll-loops -ftree-vectorize

all:
	$(CXX) $(FLAGS) NativeTree.h NativeTreeBest.cpp testNativeTree.cpp -o testNativeTreeBest
	$(CXX) $(FLAGS) NativeTree.h NativeTreeWorst.cpp testNativeTree.cpp -o testNativeTreeWorst
gdb:
	$(CXX) $(FLAGS) NativeTree.h NativeTreeBest.cpp testNativeTree.cpp -o testNativeTreeBest -g
	$(CXX) $(FLAGS) NativeTree.h NativeTreeWorst.cpp testNativeTree.cpp -o testNativeTreeWorst -g

assembly:
	@objdump -d -S testNativeTreeBest > best.as
	@objdump -d -S testNativeTreeWorst > worst.as
	vim -d best.as worst.as
clean:
	@rm *.as
