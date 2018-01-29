#/bin/bash

for d in ./*/; do
	cd $d
	echo "Training $d"
	./trainForest.py > train.txt
	cd ..
done