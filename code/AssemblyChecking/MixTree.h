#ifndef MIXTREE_H
#define MIXTREE_H

bool MixTree_predict(short const pX[1]);
bool MixTree_predict0(short const pX[1]);
bool MixTree_predictRest0(short const pX[1], unsigned short subroot);

struct MixTree_Node0 {
                        bool isLeaf;
                        bool prediction;
                        unsigned char feature;
                        unsigned short threshold;
                        unsigned short leftChild;
                        unsigned short rightChild;
                };
#endif