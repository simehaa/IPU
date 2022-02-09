POPC ?= popc
PROTOC ?= protoc

LDLIBS ?= -lpoplar -lboost_program_options
SOURCES = main.cpp
TARGETS = main

CXX = g++-11
CPPFLAGS ?= -std=c++20
CXXFLAGS ?= -O3
POPXXFLAGS ?= -O3 -target=ipu2

all: $(TARGETS)

main: main.o
	$(CXX) $(LDFLAGS) $+ $(LOADLIBES) $(LDLIBS) -o $@

main.o: codelets.gp main.cpp utils.hpp
main.cpp: utils.hpp

%.gp: %.cpp
	$(POPC) $(POPXXFLAGS) $+ -o $@

.PHONY: clean
clean:
	$(RM) $(TARGETS) *.o *.gp
