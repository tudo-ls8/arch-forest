CXX = g++
FLAGS = -std=c++11 -Wall -O0 -funroll-loops -ftree-vectorize

all:
	$(CXX) $(FLAGS) NativeTree.h NativeTree.cpp testNativeTree.cpp -o testNativeTree
	$(CXX) $(FLAGS) IfTree.h IfTree.cpp testIfTree.cpp -o testIfTree
swap:
	$(CXX) $(FLAGS) IfTree.h swap_IfTree.cpp testIfTree.cpp -o testIfTreeSwap
	$(CXX) $(FLAGS) IfTree.h origin_IfTree.cpp testIfTree.cpp -o testIfTreeOri
gdb:
	$(CXX) $(FLAGS) NativeTree.h NativeTree.cpp testNativeTree.cpp -o testNativeTree -g
	$(CXX) $(FLAGS) IfTree.h IfTree.cpp testIfTree.cpp -o testIfTree -g

assembly-if: testIfTreeOri testIfTreeSwap
	objdump -d -S testIfTreeOri > IftreeOri.as
	objdump -d -S testIfTreeSwap > IftreeSwap.as

assembly-native: testNativeTree 
	objdump -d -S testNativeTree > Native.as
clean:
	@rm *.as
