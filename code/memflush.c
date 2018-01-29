#include<stdlib.h> 
int main() {
     const int size = 300*1024; // Allocate 20M. Set much larger then L2
     char *c = (char *)malloc(size);
     for (int i = 0; i < 0xffff; i++)
       for (int j = 0; j < size; j++)
         c[j] = i*j;
 }
