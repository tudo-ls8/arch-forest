#/bin/bash

for d in ./*/; do
	echo "Generating $d"
	./generateCode.py $d
done
