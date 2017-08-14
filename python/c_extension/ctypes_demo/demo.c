#include <math.h>

int gcd(int x, int y){
	int g = y;
	while (x>0){
		g = x;
		x = y%x;
		y = g;
	}
	return g;
}
