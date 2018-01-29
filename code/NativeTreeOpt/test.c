#include<stdio.h>
int main(){
    int i = 0, q=0, x=0, y=0;
    printf("0\n");
    for(i=1; i<14; i++){
        q = 1 + ((i*1024*32 - 1) / 10);
        printf("%d\n", (int)q);
    }
    // leftchild of the targeted element
    for(i=1; i<14; i++){
        q = 1 + ((i*1024*32 - 1) / 10);
        printf("%d\n", (int)q*2+1);
    }
    return 0;
}
