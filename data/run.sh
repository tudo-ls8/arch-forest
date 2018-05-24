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
	sklearn=$(./runSKLearn.py $1)
	echo $d
	cd $d
	if [ "$2" == "arm" ]; then
		measurments="$d,$sklearn,\
			$(./testStandardIfTree),$(stat --printf="%s" testStandardIfTree),\
			$(./testOptimizedPathIfTree_32000),$(stat --printf="%s" testOptimizedPathIfTree_32000),\
			$(./testOptimizedPathIfTree_64000),$(stat --printf="%s" testOptimizedPathIfTree_64000),\
			$(./testStandardNativeTree),$(stat --printf="%s" testStandardNativeTree),\
			$(./testOptimizedNativeTree_8),$(stat --printf="%s" testOptimizedNativeTree_8),\
			$(./testOptimizedNativeTree_32),$(stat --printf="%s" testOptimizedNativeTree_32),\
      $(./testOptimizedNativeForest_25),$(stat --printf="%s" testOptimizedNativeForest_25)"
	else
		measurments="$d,$sklearn,\
			$(./testStandardIfTree),$(stat --printf="%s" testStandardIfTree),\
			$(./testOptimizedPathIfTree_128000),$(stat --printf="%s" testOptimizedPathIfTree_128000),\
			$(./testOptimizedPathIfTree_384000),$(stat --printf="%s" testOptimizedPathIfTree_384000),\
			$(./testStandardNativeTree),$(stat --printf="%s" testStandardNativeTree),\
			$(./testOptimizedNativeTree_25),$(stat --printf="%s" testOptimizedNativeTree_25)\
      $(./testOptimizedNativeForest_25),$(stat --printf="%s" testOptimizedNativeForest_25)"
	fi
  

	#measurments="$d,$(./testStandardIfTree),$(./testOptimizedSwapIfTree),$(./testOptimizedPathIfTree),$(./testOptimizedNodeIfTree),$(./testStandardNativeTree),$(./testOptimizedNativeTree)"
	#measurments="$d,$(./testStandardIfTree),$(./testOptimizedPathIfTree_16000),$(./testOptimizedNodeIfTree_16000),$(./testOptimizedSwapIfTree_16000),$(./testOptimizedPathIfTree_32000),$(./testOptimizedNodeIfTree_32000),$(./testOptimizedSwapIfTree_32000),$(./testOptimizedPathIfTree_64000),$(./testOptimizedNodeIfTree_64000),$(./testOptimizedSwapIfTree_64000),$(./testStandardNativeTree),$(./testOptimizedNativeTree_5),$(./testOptimizedNativeTree_10),$(./testOptimizedNativeTree_25),$(./testOptimizedNativeTree_50)"
	printf "$measurments \n"
	cd ../../../..
	#echo $measurments >> results_$2.csv
done
