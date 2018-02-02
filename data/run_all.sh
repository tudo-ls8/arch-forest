#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

echo "config,Mean-StandardIfTree,Var-StandardIfTree,Mean-OptimizedIfTree, Var-OptimizedIfTree,Mean-StandardNativeTree,Var-StandardNativeTree,Mean-OptimizedNativeTree,Var-OptimizedNativeTree" > results_$1.txt
for d in ./*/; do
	echo "Profiling $d"
	./run.sh $d $1 >> results_$1.txt
done
