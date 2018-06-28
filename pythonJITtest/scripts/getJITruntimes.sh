#/bin/bash

for d in */; do
	echo "$d"
	cd "$d"	
	python OptNativePyForest.py
	cd ..
done
