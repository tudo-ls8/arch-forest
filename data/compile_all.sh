#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

for d in */; do
	echo "Compiling $d"
	./compile.sh $d $1
done
