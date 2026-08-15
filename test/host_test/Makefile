# Host test harness for velvet_reverb.
#
# Each test is one .c file that includes host_shim.h and links the real
# velvet_reverb.c with VELVET_REVERB_HOST=1, so what is measured is the
# firmware source itself rather than a model of it.
#
#   make run          the correctness harness (silence, impulse, IR, relocation)
#   make <test>       build one test, e.g. make hf_decay
#   make <test>-run   build and run it
#
# host_main is built with ASan + UBSan: it is the memory-safety and correctness
# gate, and it renders little enough audio to afford them. The measurement tests
# render seconds to minutes of audio per run and are looking at numbers rather
# than at memory, so they are built -O2 without sanitizers.

CC      ?= clang
WARN    := -std=c99 -Wall -Wextra -Wno-unused-parameter -Wno-unknown-attributes
BASE    := $(WARN) -DVELVET_REVERB_HOST=1 -I. -I../..
CFLAGS  := $(BASE) -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS := -fsanitize=address,undefined
TESTCF  := $(BASE) -O2
LDLIBS  := -lm

REVERB  := ../../velvet_reverb.c
HDRS    := host_shim.h ../../velvet_reverb.h
TARGET  := test_velvet

# Every test except the sanitized harness.
TESTS   := $(basename $(filter-out host_main.c, $(wildcard *.c)))

# The tests themselves must NOT be phony: they are real files built by the
# pattern rule below, and a phony target does not consider implicit rules.
.PHONY: all run clean $(addsuffix -run, $(TESTS))

all: $(TARGET)

$(TARGET): host_main.c $(REVERB) $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ host_main.c $(REVERB) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

%: %.c $(REVERB) $(HDRS)
	$(CC) $(TESTCF) -o $@ $< $(REVERB) $(LDLIBS)

%-run: %
	./$<

clean:
	rm -f $(TARGET) $(TESTS)
	rm -rf *.dSYM
