BUILD_FLAGS = -O2 -fno-pic -ggdb -c
LINK_FLAGS = -static -fno-pic -v

a.out : a.o b.o main.o
	gcc ${LINK_FLAGS} a.o b.o main.o

a.o: a.c
	gcc ${BUILD_FLAGS} a.c

b.o: b.c
	gcc ${BUILD_FLAGS} b.c

main.o: main.c
	gcc ${BUILD_FLAGS} main.c

clean:
	rm *.o *.out