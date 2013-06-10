COMPILER = g++
DFLAGS = -ansi -pedantic -Wall
CFLAGS = -c
OFLAGS = -o
EXECNAME = main

suffixtree:	main.o suffix_tree.o
	${COMPILER} ${DFLAGS} main.o suffix_tree.o ${OFLAGS} ${EXECNAME}

suffix_tree.o:	suffix_tree.c suffix_tree.h
	${COMPILER} ${DFLAGS} ${CFLAGS} suffix_tree.c

main.cpp:	suffix_tree.h 
	${COMPILER} ${DFLAGS} ${CFLAGS} main.cpp 

clean: 
	rm *.o 
	rm ${EXECNAME}
