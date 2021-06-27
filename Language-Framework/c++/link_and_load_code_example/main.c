#include <stdio.h>

extern int x, y;
int foo(int a, int b);

int main() {
    printf("%d + %d = %d\n", x, y, foo(x, y));
}