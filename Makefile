# The compiler: gcc for C program, define as g++ for C++
CC=g++

# Compiler flgs
CFLAGS=-g -Wall

# Source directory
SRC=src

# Program directory
BIN=bin

# Open CV libraries
OPENCV=`pkg-config opencv --cflags --libs`
LIBS=$(OPENCV)

all:	$(BIN)/selectsearch	$(BIN)/houghtransform	$(BIN)/maskrcnn

$(BIN)/selectsearch:	$(SRC)/selectsearch.cpp
	$(CC) $(CFLAGS) -o $(BIN)/selectsearch $(SRC)/selectsearch.cpp $(LIBS)

$(BIN)/houghtransform:	$(SRC)/houghtransform.cpp
	$(CC) $(CFLAGS) -o $(BIN)/houghtransform $(SRC)/houghtransform.cpp $(LIBS)

$(BIN)/maskrcnn:        $(SRC)/maskrcnn.cpp
	$(CC) $(CFLAGS) -o $(BIN)/maskrcnn $(SRC)/maskrcnn.cpp $(LIBS)

run:
	./$(BIN)/selectsearch
	./$(BIN)/houghtransform
	./$(BIN)/maskrcnn

runSearch:
	./$(BIN)/selectsearch

runHough:
	./$(BIN)/houghtransform

runMask:
	./$(BIN)/maskrcnn

clean:
	$(RM) $(BIN)/* $(SRC)/*.o
