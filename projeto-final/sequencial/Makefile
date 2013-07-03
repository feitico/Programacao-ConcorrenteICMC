############### Makefile ###############
all: projeto
projeto: jacobi.o main.o
	g++ -o jacobi jacobi.o main.o

jacobi.o: jacobi.cpp
	g++ -o jacobi.o -c jacobi.cpp -W -Wall -ansi -pedantic
main.o: main.cpp jacobi.h
	gcc -o main.o -c main.cpp -W -Wall -ansi -pedantic

clean:
	rm -rf jacobi *.o
