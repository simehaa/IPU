POPC ?= popc
PROTOC ?= protoc

LDLIBS ?= -lpoplar -lpoputil -lpopops -lboost_program_options -lprotobuf
SOURCES = main.cpp
TARGETS = main

CPPFLAGS ?= -std=c++11
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
