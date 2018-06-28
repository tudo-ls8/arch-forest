from testStruct import * # imports pX
#from treeStruct import * # imports nodePos and tree

import time
import cProfile as profile

from llvmlite import ir
from numba import jit
from numba import njit

import numpy as np

pXlen = len(pX)

def main():
    # burn-in
    print(pXlen)

    for m in range (2):
        for n in range(pXlen):
            OptPathIfPredict(pX[n])

    profile.run("timePrediction()")

def timePrediction():
    for p in range(pXlen):
        OptPathIfPredict(pX[p])

@jit
def OptPathIfPredict(pX):
    predCnt = np.array([0,0,0,0,0,0,0])
    if(pX[7] > 0.992840051651001):
        if(pX[4] > 0.04450000077486038):
            if(pX[10] <= 10.016666412353516):
                if(pX[9] <= 0.574999988079071):
                    if(pX[1] > 0.2849999964237213):
                        predCnt[2]+=1
                    else :
                        predCnt[3]+=1

                else :
                    if(pX[5] <= 40.5):
                        predCnt[2]+=1
                    else :
                        predCnt[3]+=1


            else :
                if(pX[2] <= 0.33500000834465027):
                    if(pX[5] <= 50.0):
                        predCnt[3]+=1
                    else :
                        predCnt[3]+=1

                else :
                    if(pX[9] <= 0.7549999952316284):
                        predCnt[3]+=1
                    else :
                        predCnt[4]+=1



        else :
            if(pX[10] <= 10.75):
                if(pX[2] > 0.19499999284744263):
                    if(pX[5] > 24.5):
                        predCnt[3]+=1
                    else :
                        predCnt[2]+=1

                else :
                    if(pX[1] > 0.1850000023841858):
                        predCnt[2]+=1
                    else :
                        predCnt[3]+=1


            else :
                if(pX[0] <= 7.149999618530273):
                    if(pX[9] <= 0.4699999988079071):
                        predCnt[3]+=1
                    else :
                        predCnt[3]+=1

                else :
                    if(pX[2] <= 0.48000001907348633):
                        predCnt[3]+=1
                    else :
                        predCnt[4]+=1




    else :
        if(pX[3] > 1.75):
            if(pX[7] <= 0.9922149777412415):
                if(pX[6] <= 158.5):
                    if(pX[9] <= 0.5649999976158142):
                        predCnt[4]+=1
                    else :
                        predCnt[4]+=1

                else :
                    if(pX[1] <= 0.3050000071525574):
                        predCnt[3]+=1
                    else :
                        predCnt[3]+=1


            else :
                if(pX[3] > 3.674999952316284):
                    if(pX[8] > 2.9650001525878906):
                        predCnt[3]+=1
                    else :
                        predCnt[2]+=1

                else :
                    if(pX[4] <= 0.045500002801418304):
                        predCnt[2]+=1
                    else :
                        predCnt[3]+=1



        else :
            if(pX[7] <= 0.9917700290679932):
                if(pX[3] > 1.0499999523162842):
                    if(pX[1] <= 0.6299999952316284):
                        predCnt[3]+=1
                    else :
                        predCnt[5]+=1

                else :
                    if(pX[8] > 3.069999933242798):
                        predCnt[3]+=1
                    else :
                        predCnt[3]+=1


            else :
                if(pX[10] > 10.350000381469727):
                    if(pX[2] <= 0.41499999165534973):
                        predCnt[2]+=1
                    else :
                        predCnt[3]+=1

                else :
                    if(pX[1] <= 0.2750000059604645):
                        predCnt[3]+=1
                    else :
                        predCnt[2]+=1





    pred = 0
    cnt = predCnt[0]
    for i in range(7):
        if (predCnt[i] > cnt):
            cnt = predCnt[i]
            pred = i

    return pred

if __name__ == "__main__":
   main()
