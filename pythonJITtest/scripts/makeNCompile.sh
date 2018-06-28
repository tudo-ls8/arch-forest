#/bin/bash

for d in */; do
	cd "$d"	
	make
	./testOptimizedNativeForest_25
	cd ..
done
