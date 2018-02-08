#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

if [ "$1" == "arm" ]; then
	echo "config,sklearn,Mean-testStandardIfTree,Var-testStandardIfTree,Filesize-testStandardIfTree,Mean-testOptimizedPathIfTree_32000,Var-testOptimizedPathIfTree_32000,Filesize-testOptimizedPathIfTree_32000,Mean-testOptimizedPathIfTree_64000,Var-testOptimizedPathIfTree_64000,Filesize-testOptimizedPathIfTree_64000,Mean-testStandardNativeTree,Var-testStandardNativeTree,Filesize-testStandardNativeTree,Mean-testOptimizedNativeTree_8,Var-testOptimizedNativeTree_8,Filesize-testOptimizedNativeTree_8,Mean-testOptimizedNativeTree_32,Var-testOptimizedNativeTree_32,Filesize-testOptimizedNativeTree_32" > results_$1.txt
else
	echo "config,sklearn,Mean-testStandardIfTree,Var-testStandardIfTree,Filesize-testStandardIfTree,Mean-testOptimizedPathIfTree_128000,Var-testOptimizedPathIfTree_128000,Filesize-testOptimizedPathIfTree_128000,Mean-testOptimizedPathIfTree_384000,Var-testOptimizedPathIfTree_38464000,Filesize-testOptimizedPathIfTree_384000,Mean-testStandardNativeTree,Var-testStandardNativeTree,Filesize-testStandardNativeTree,Mean-testOptimizedNativeTree_25,Var-testOptimizedNativeTree_25,Filesize-testOptimizedNativeTree_25" > results_$1.txt
fi

for d in ./*/; do
	echo "Profiling $d"
	./run.sh $d $1 >> results_$1.txt
done
