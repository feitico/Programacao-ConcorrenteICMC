CC=gcc
CFLAGS=-fopenmp -Wall -ansi
LDFLAGS=-fopenmp
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)
EXEC=hello

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC) it*.txt
