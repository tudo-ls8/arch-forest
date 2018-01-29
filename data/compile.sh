#/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

cd $1/cpp

for d in ./*/; do
	echo $d
	cd $d
	
	mkdir -p arm
	make arm-cpu
	
	mkdir -p intel
	make intel-cpu

	cd ..
done

#	mkdir -p arm
#	make arm-cpu
#
#	mkdir -p intel
#	make intel-cpu
#
#	cd ..
#done
