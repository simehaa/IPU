POPC ?= popc
PROTOC ?= protoc

LDLIBS ?= -lpoplar -lpoputil -lpopops -lboost_program_options -lprotobuf
SOURCES = main.cpp
TARGETS = main

CPPFLAGS ?= -std=c++11
CXXFLAGS ?= -O3
POPXXFLAGS ?= -O3 -target=ipu2

all: $(TARGETS)

main: main.o image.pb.o
	$(CXX) $(LDFLAGS) $+ $(LOADLIBES) $(LDLIBS) -o $@

main.o: codelets.gp utils.hpp
main.cpp: utils.hpp
utils.hpp: image.pb.h

%.gp: %.cpp
	$(POPC) $(POPXXFLAGS) $+ -o $@

%.pb.cc %.pb.h %_pb2.py: %.proto
	$(PROTOC) --cpp_out=. --python_out=. $+

%.jpg.bin: %.jpg image_pb2.py
	$(PYTHON) ./image_to_bin.py --in-file=$<

.PRECIOUS: %_pb2.py
.PHONY: clean
clean:
	$(RM) $(TARGETS)
	$(RM) *.o
	$(RM) *.gp
	$(RM) *.pb.*
	$(RM) *_pb2.py
