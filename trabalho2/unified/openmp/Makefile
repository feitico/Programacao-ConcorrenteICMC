CC=g++
CFLAGS=-fopenmp -Wall -ansi
LDFLAGS=-fopenmp
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)
EXEC=$(SRC:.cpp=)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC)
