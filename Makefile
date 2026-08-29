######################################################################
# ternary-moe build
#
#   make            build lib + all apps
#   make test       build & run every tests/test_*.c, print a summary
#   make bench      build & run every bench/bench_*.c, print a summary
#   make all        build lib + apps + tests + benches
#   make clean      remove build/ and bin/
#
# Override CC / CFLAGS on the command line if needed, e.g. on a
# Termux/Android box without -march=native support:
#   make CFLAGS_ARCH=
######################################################################

CC       ?= cc
SRC_DIR  := src
APP_DIR  := app
TEST_DIR := tests
BENCH_DIR:= bench
BUILD    := build
BIN      := bin

# -march=native isn't always available (older gcc, some cross setups),
# so probe for it once instead of hard-failing the whole build.
CFLAGS_ARCH ?= $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native)

WARN     := -Wall -Wextra
OPT      := -O3 -funroll-loops -fomit-frame-pointer $(CFLAGS_ARCH)
STD      := -std=c11 -D_POSIX_C_SOURCE=199309L
CFLAGS   ?= $(STD) $(WARN) $(OPT) -I$(SRC_DIR)
LDLIBS   := -lm

SRCS     := $(wildcard $(SRC_DIR)/*.c)
OBJS     := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/obj/%.o,$(SRCS))
LIB      := $(BUILD)/libmoe.a

APP_SRCS   := $(wildcard $(APP_DIR)/*.c)
APP_BINS   := $(patsubst $(APP_DIR)/%.c,$(BIN)/%,$(APP_SRCS))

TEST_SRCS  := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS  := $(patsubst $(TEST_DIR)/%.c,$(BUILD)/tests/%,$(TEST_SRCS))

BENCH_SRCS := $(wildcard $(BENCH_DIR)/*.c)
BENCH_BINS := $(patsubst $(BENCH_DIR)/%.c,$(BUILD)/bench/%,$(BENCH_SRCS))

.PHONY: all lib apps test bench clean
.DEFAULT_GOAL := apps

lib: $(LIB)

apps: lib $(APP_BINS)

all: apps test bench

$(BUILD)/obj/%.o: $(SRC_DIR)/%.c | $(BUILD)/obj
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS) | $(BUILD)
	$(AR) rcs $@ $(OBJS)

$(BIN)/%: $(APP_DIR)/%.c $(LIB) | $(BIN)
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/tests/%: $(TEST_DIR)/%.c $(LIB) | $(BUILD)/tests
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/bench/%: $(BENCH_DIR)/%.c $(LIB) | $(BUILD)/bench
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/obj $(BUILD)/tests $(BUILD)/bench $(BUILD) $(BIN):
	mkdir -p $@

test: $(TEST_BINS)
	@bash scripts/run_tests.sh $(TEST_BINS)

bench: $(BENCH_BINS)
	@bash scripts/run_bench.sh $(BENCH_BINS)

clean:
	rm -rf $(BUILD) $(BIN)
