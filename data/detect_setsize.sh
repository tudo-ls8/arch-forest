#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

echo "Training forest"

cd wearable-body-postures
./trainForest.py
cd ..

echo "setsize,config,Mean-StandardIfTree,Var-StandardIfTree,Mean-OptimizedIfTree, Var-OptimizedIfTree,Mean-StandardNativeTree,Var-StandardNativeTree,Mean-OptimizedNativeTree,Var-OptimizedNativeTree" > results.txt
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14
do
	echo -e "\tGenerating Code for setsize $i"
	./generateCode.py wearable-body-postures $1 $i
	echo -e "\tCompiling code for setsize $i"
	./compile.sh wearable-body-postures $1
	echo -e "\tProfiling code for setsize $i"

	echo "Profiling forest"
	./run.sh wearable-body-postures $1 | sed -e "s/^/$i,/" >> results.txt
done
