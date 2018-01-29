#Json generation should be done
#generate.py for code should be done
#gcc -O0 -std=c99 memflush.c 	
sync
#./a.out
sudo perf stat --repeat 10 -e branch-misses,branch-instructions,L1-dcache-loads-misses,L1-dcache-load,L1-icache-load-misses,instructions,cycles ./testNativeTreeBest 
sync
#./a.out
sudo perf stat --repeat 10 -e branch-misses,branch-instructions,L1-dcache-loads-misses,L1-dcache-load,L1-icache-load-misses,instructions,cycles ./testNativeTreeWorst 
