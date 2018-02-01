#/bin/bash

for d in ./*/; do
	if ["$d" -ne "synthetic-chain"] 
	then
		echo "Generating $d"
		./generateCode.py $d
	fi
done
