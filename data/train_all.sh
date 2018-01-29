#/bin/bash

<<<<<<< HEAD
for d in ./*/; do
	cd $d
	echo "Training $d"
	./trainForest.py > train.txt
	cd ..
done
=======
for d in */; do
	cd d
	echo "Training $d"
	./trainForest.py > train.txt
done
>>>>>>> 3df02548914e5b679979f479fd0e24b66b2f8ba8
