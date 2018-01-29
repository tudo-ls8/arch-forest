#/bin/bash

for d in */; do
	echo "Compiling $d"
	./compile $d
done