#include <iostream>
int OptimizedNativeForest_25_predict(float const pX[784]);
struct OptimizedNativeForest_25_Node {
                    //bool isLeaf;
                    //unsigned int prediction;
                    unsigned short feature;
                    float split;
                    unsigned int leftChild;
                    unsigned int rightChild;
                    unsigned char indicator;

            };
