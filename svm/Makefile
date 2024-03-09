CXX ?= g++

CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)
EXEC = main
OBJS = main.o MultipleImageWindow.o

all: main

main: main.o MultipleImageWindow.o
	 $(CXX) -o main main.o MultipleImageWindow.o $(LDFLAGS)

MultipleImageWindow.o: MultipleImageWindow.cpp MultipleImageWindow.h
	$(CXX) $(CXXFLAGS) MultipleImageWindow.cpp

main.o: main.cpp MultipleImageWindow.h
	$(CXX) $(CXXFLAGS) main.cpp

clean: ; rm -f main.o main MultipleImageWindow.o
