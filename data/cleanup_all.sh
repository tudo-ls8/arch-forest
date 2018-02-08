#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

for d in ./*/; do
	echo "Cleaning ./$d/cpp/$1"
	rm -r $d/cpp/$1 

	echo "Cleaning ./$d/text/"
	rm $d/text/* 
done