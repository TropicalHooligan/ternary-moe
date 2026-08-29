######################################################################
# ternary-moe build - Optimized for cross-platform (x86_64, ARM64, Termux)
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
# For ARM64 with NEON support:
#   make ARCH_FLAGS="-mcpu=neoverse-v1 -mtune=neoverse-v1"
######################################################################

CC       ?= cc
SRC_DIR  := src
APP_DIR  := app
TEST_DIR := tests
BENCH_DIR:= bench
BUILD    := build
BIN      := bin

# Architecture detection and flags
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)

# Default architecture flags
CFLAGS_ARCH ?= $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native)

# ARM64/NEON optimization flags
ifeq ($(UNAME_M),aarch64)
  # Check if compiler supports NEON
  NEON_SUPPORT := $(shell $(CC) -mfpu=neon -E -x c /dev/null >/dev/null 2>&1 && echo -mfpu=neon)
  CFLAGS_ARCH += $(NEON_SUPPORT)
  # Try to detect best ARM CPU
  CPU_FLAGS := $(shell $(CC) -mcpu=native -E -x c /dev/null >/dev/null 2>&1 && echo -mcpu=native)
  CFLAGS_ARCH += $(CPU_FLAGS)
endif

# Common optimization flags
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
OPT      := -O3 -funroll-loops -fomit-frame-pointer -ffast-math -fno-finite-math-only
STD      := -std=c11 -D_POSIX_C_SOURCE=199309L
CFLAGS   ?= $(STD) $(WARN) $(OPT) $(CFLAGS_ARCH) -I$(SRC_DIR)
LDLIBS   := -lm

# Additional security/performance flags
CFLAGS   += -fstrict-aliasing -fno-strict-overflow

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

# Display build info
info:
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "UNAME_M: $(UNAME_M)"
	@echo "CFLAGS_ARCH: $(CFLAGS_ARCH)"
