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
	measurments="$d,$(./testStandardIfTree),$(./testOptimizedIfTree),$(./testStandardNativeTree),$(./testOptimizedNativeTree)"
	printf "$measurments \n"
	cd ../../../..
	#echo $measurments >> results_$2.csv
done
