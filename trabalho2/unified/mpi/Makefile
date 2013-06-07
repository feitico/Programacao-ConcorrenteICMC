CC=mpicxx
CFLAGS=-Wall -ansi
SRC=$(wildcard *.cpp)
EXEC=$(SRC:.cpp=)

all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) -o $@ $(SRC) $(CFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC)
