CC=mpicc
CFLAGS=-lm
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.c)
EXEC=$(SRC:.c=)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(CFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC) it*.txt
