#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

echo "config,Mean-testStandardIfTree,Var-testStandardIfTree,Mean-testOptimizedPathIfTree_16000,Var-testOptimizedPathIfTree_16000,Mean-testOptimizedNodeIfTree_16000,Var-testOptimizedNodeIfTree_16000,Mean-testOptimizedSwapIfTree_16000,Var-testOptimizedSwapIfTree_16000,Mean-testOptimizedPathIfTree_32000,Var-testOptimizedPathIfTree_32000,Mean-testOptimizedNodeIfTree_32000,Var-testOptimizedNodeIfTree_32000,Mean-testOptimizedSwapIfTree_32000,Var-testOptimizedSwapIfTree_32000,Mean-testOptimizedPathIfTree_64000,Var-testOptimizedPathIfTree_64000,Mean-testOptimizedNodeIfTree_64000,Var-testOptimizedNodeIfTree_64000,Mean-testOptimizedSwapIfTree_64000,Var-testOptimizedSwapIfTree_64000,Mean-testStandardNativeTree,Var-testStandardNativeTree,Mean-testOptimizedNativeTree_5,Var-testOptimizedNativeTree_5,Mean-testOptimizedNativeTree_10,Var-testOptimizedNativeTree_10,Mean-testOptimizedNativeTree_25,Var-testOptimizedNativeTree_25,Mean-testOptimizedNativeTree_50,Var-testOptimizedNativeTree_50" > results_$1.txt
for d in ./*/; do
	echo "Profiling $d"
	./run.sh $d $1 >> results_$1.txt
done
