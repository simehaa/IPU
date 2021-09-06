POPC ?= popc
POPXXFLAGS ?= -target=ipu2 -O3

CXXFLAGS ?= -O3
LDLIBS ?= -lpoplar

ALL = main codelets.gp

all: $(ALL)

%.gp: %.cpp
	$(POPC) $(POPXXFLAGS) $+ -o $@

.PHONY: clean
clean:
	$(RM) $(ALL)
