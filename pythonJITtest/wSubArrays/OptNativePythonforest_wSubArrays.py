from testStruct import * #imports pX
from treeStruct import * # imports nodePos and tree

import time
import cProfile as profile

from llvmlite import ir
from numba import jit
from numba import njit

#tree[i][feature, split, lch, rch, ind]
#tree[i][0,       1,      2,   3,  4  ]

pXlen = len(pX)
nrTrees = len(nodePos)

def main():
    # burn-in
    print(pXlen)

    for m in range (2):
        for n in range(pXlen):
            OptNativePyForestPredict(pX[n])

    profile.run("timePrediction()")

def timePrediction():
    #print(sum(isinstance(i, list) for i in pX))
    for p in range(pXlen):
        OptNativePyForestPredict(pX[p])
    #print(OptNativePyForestPredict(pX))

#@jit
def OptNativePyForestPredict(pX):
    predCnt = np.array([0,0,0,0,0,0,0])
    for treeIndex in range(nrTrees):
        # index for node
        i = nodePos[treeIndex]
        while(True):
            # change index for nodes to index for array
            #i = i*5
            if (pX[int(tree[i,0])] <= tree[i,1]):
                if (tree[i,4] == 0 or tree[i,4] == 2):
                    i = int(tree[i,2])
                else:
                    predCnt[int(tree[i,2])] += 1
                    break
            else:
                if (tree[i,4] == 0 or tree[i,4] == 1):
                    i = int(tree[i,3])
                else:
                    predCnt[int(tree[i,3])] += 1
                    break

    pred = 0
    cnt = predCnt[0]
    for i in range(nrTrees):
        if (predCnt[i] > cnt):
            cnt = predCnt[i]
            pred = i

    return pred

if __name__ == "__main__":
   main()
