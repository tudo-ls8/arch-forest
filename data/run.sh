#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

if [ "$#" -lt 2 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

#echo "config,Mean-StandardIfTree,Var-StandardIfTree,Mean-OptimizedIfTree, Var-OptimizedIfTree,Mean-StandardNativeTree,Var-StandardNativeTree,Mean-OptimizedNativeTree,Var-OptimizedNativeTree"
#echo "$1/cpp/$2"

for d in ./$1/cpp/$2/*/; do
    echo $d
	cd $d
	#measurments="$d,$(./testStandardIfTree),$(./testOptimizedSwapIfTree),$(./testOptimizedPathIfTree),$(./testOptimizedNodeIfTree),$(./testStandardNativeTree),$(./testOptimizedNativeTree)"
	measurments="$d,$(./testStandardIfTree),$(./testOptimizedPathIfTree_16000),$(./testOptimizedNodeIfTree_16000),$(./testOptimizedSwapIfTree_16000),$(./testOptimizedPathIfTree_32000),$(./testOptimizedNodeIfTree_32000),$(./testOptimizedSwapIfTree_32000),$(./testOptimizedPathIfTree_64000),$(./testOptimizedNodeIfTree_64000),$(./testOptimizedSwapIfTree_64000),$(./testStandardNativeTree),$(./testOptimizedNativeTree_5),$(./testOptimizedNativeTree_10),$(./testOptimizedNativeTree_25),$(./testOptimizedNativeTree_50)"
	printf "$measurments \n"
	cd ../../../..
	#echo $measurments >> results_$2.csv
done
