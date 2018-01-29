#include <stdio.h>

int main(int argc, char *argv[])
{
	int reg;
	if(argc<1){
		printf("Missing argument. Enter 1 to enable RoundRobin, anything else to disable it");
		return -1;
	}
	__asm__("MRC p15, 0, register, c1, c0, 0");
	if (argv[0]=="1"){
		reg = (reg | 100000000000000);
	}
	else{
		reg = (reg & 011111111111111);
	}
	__asm__("MCR p15, 0, register, c1, c0, 0");
	__asm__("MCR p15, 0, SBZ, c7, c5, 4");
	
	return 0;
}
