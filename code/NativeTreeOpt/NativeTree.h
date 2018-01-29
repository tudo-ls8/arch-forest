#ifndef NATIVETREE_H
#define NATIVETREE_H

bool NativeTree_predict(short const pX[1]);
struct NativeTree_Node0 {
			bool isLeaf;
			bool prediction;
			unsigned char feature;
			unsigned short threshold;
			unsigned short leftChild;
			unsigned short rightChild;
		};
bool NativeTree_predict0(short const pX[1]);
#endif
