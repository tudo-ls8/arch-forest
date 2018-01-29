#/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

if [ "$#" -ne 2 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

cd $1

for d in cpp/*/; do
	cd $d/$2

	sync	
	perf stat -o performance.txt --repeat 10 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testOptimizedIfTree

	sync	
	perf stat -o performance.txt --append --repeat 10 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testStandardIfTree

	sync	
	perf stat -o performance.txt --append --repeat 10 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testOptimizedNativeTree

	sync	
	perf stat -o performance.txt --append --repeat 10 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testStandardNativeTree

	cd ../..
done

# echo "run, prob, size, OptimizedIfTree,StandardIfTree,OptimizedNativeTree,StandardNativeTree" > results.csv
# for d in cpp/*/; do
# 	cd $d
# 	config=$(echo $d | sed s/cpp//g | sed s/p_//g | sed s,/,,g | sed s/_s_/,/g )
# 	measurment=$(grep "seconds" performance.txt | cut  -d ' ' -f 8 | sed s/,/./g | paste -d "," - - | paste -d "," - -)
	
# 	cd ../..
# 	echo -n $d >> results.csv
# 	echo -n "," >> results.csv
# 	echo -n $config >> results.csv
# 	echo -n "," >> results.csv
# 	echo $measurment >> results.csv
# done

#Json generation should be done
#generate.py for code should be done
#gcc -O0 -std=c99 memflush.c 	

#echo 3 > /proc/sys/vm/drop_caches
#./a.out
#perf stat --repeat 5 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testNativeTree 
#sync
#echo 3 > /proc/sys/vm/drop_caches
#./a.out
#perf stat --repeat 5 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testMixTree 
#sync
#echo 3 > /proc/sys/vm/drop_caches
#./a.out
#perf stat --repeat 5 -e cache-misses,L1-dcache-stores,L1-dcache-load-misses,L1-icache-prefetches,L1-icache-load-misses,instructions,cycles ./testIfTree 