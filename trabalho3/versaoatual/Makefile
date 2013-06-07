CC=mpicxx
CFLAGS=-fopenmp -Wall -ansi
LDFLAGS=$(CFLAGS)
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)
EXEC=$(SRC:.cpp=)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -rf $(OBJ) $(EXEC)

