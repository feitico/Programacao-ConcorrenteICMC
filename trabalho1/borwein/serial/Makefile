CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=-lgmp
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)
EXEC=borwein

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC) it*.txt
