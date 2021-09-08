CC = gcc
CFLAGS = -O3 -fopenmp
TARGETS = main
OBJECTS = main.o

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $+ -o $@ 

clean:
	$(RM) *.o
	$(RM) $(TARGETS)