# Size of Node prediction by batch method  

There are two files in our branch. These files are different methods to assign the size of node in IfTreeConventer.py. The "batch_IfTreeConverter.py" is the batch method. The "original_IfTreeConventer.py" is the original method with our revised code that will generate a new file allNode to help us evaluate the accuracy.  

## How to use?(take batch_IfTreeConverter.py for example) 

$ rm arch-forest/code/IfTreeConverter.py  

$ cp arch-forest/IfTreePatterns/batch_IfTreeConverter.py archforest/code/IfTreeConverter.py $ python3 generateCode.py <dataSetName> <architecture>  

## How to prove the predicted size of node is near-optimal?(Take adult/RF_20 for example) $ cd arch-forest/data  

$ ./train_all.sh $ ./generate_all.sh intel  

$ cp adult/cpp/intel/RF_20/OptimizedPathIfTree.cpp .  

$ g++ -c -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize OptimizedPathIfTree.cpp  

$ objdump -h OptimizedPathIfTree.o > actualCodeSize  

With these instructions, two files will be generated, allNode and actualCodeSize. The allNode file contains two columns in each row. The first column is tree_id and the second column is tree_size. In the actualCodeSize file, the first three column is Idx, name, and size. We can find tree_id in the name column and get the tree size in the size column(the tree_size here is in hexadecimal format). By comparing the tree size in the actualCodeSize file and allNode file, we can know the difference between the predicted size and the actual size. 
