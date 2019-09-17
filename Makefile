# the compiler: gcc for C program, define as g++ for C++
CC=g++

# compiler flgs
CFLAGS=-g -Wall

# source directory
SRC=src

# program directory
EXEC=executables

OPENCV=`pkg-config opencv --cflags --libs`
LIBS=$(OPENCV)

$(EXEC)/selectsearch:$(SRC)/selectsearch.cpp
	$(CC) $(CFLAGS) -o $(EXEC)/selectsearch $(SRC)/selectsearch.cpp $(LIBS)
run:
	./$(EXEC)/selectsearch

clean:
	$(RM) $(EXEC)/* *.o
