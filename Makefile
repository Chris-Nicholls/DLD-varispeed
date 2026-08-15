BINARYNAME = main

STARTUP = startup_stm32f427_437xx.s
SYSTEM = system_stm32f4xx.c
LOADFILE = stm32f427.ld

DEVICE = stm32/device
CORE = stm32/core
PERIPH = stm32/periph

BUILDDIR = build

# ---- Profiling firmware ---------------------------------------------------
# Answers "where does the reverb block actually spend its time", which the
# normal build cannot: its diag thresholds only log outliers, so every
# distribution is biased toward its own tail. PROFILE=1 instead logs every
# stage of one-in-128 blocks with zero thresholds — an unbiased sample — and
# adds a T2 breakdown into DMA wait / DMA kick / convolution.
#
#   make PROFILE=1            instrumented build (FSK stream on right SEND)
#   make PROFILE=1 NODMA=1    same, but T2 prefetch replaced by CPU copies
#   make PROFILE=1 flash      flash the instrumented build
#
# Build both and compare reverb_T2 to settle whether the DMA prefetch is
# earning its keep. Objects go to build_profile/ so the normal build and its
# map file are never clobbered.
# ---- Pre-T2 saturator off ---------------------------------------------------
# make BISECT=t2sat flash   builds normal, listenable firmware with the always-on
# pre-T2 tanh removed. That saturator is the dominant source of broadband hash in
# the harness (8 dB at matched loudness), so its drive is a character decision
# that has to be made by ear; this is the A/B for it.
ifeq ($(BISECT),t2sat)
BISECT_FLAGS = -DDIAG_BISECT_NO_T2SAT
BUILDDIR     = build_bisect_t2sat
endif

# ---- T2 DMA off ------------------------------------------------------------
# NODMA=1 replaces the T2 SDRAM prefetch with plain CPU copies, with no
# profiling and no FSK, so it is a normal listenable firmware. This is the single
# most valuable A/B for an artifact that no host test can reproduce: the DMA is
# the largest behavioural difference between the device and the host build (the
# host just memcpy's). If the impulses vanish here, the T2 DMA path is
# implicated; if they persist, it is exonerated and the pre-delay/SDRAM side is
# next. Costs some CPU (~1k cycles/block) but stays inside the deadline.
# (defined after the PROFILE block below, which also consumes NODMA)

PROFILE ?= 0
NODMA   ?= 0
ifeq ($(PROFILE),1)
PROFILE_FLAGS  = -DDIAG_FSK_ENABLE -DDIAG_REVERB_PROFILE
# 127, not 128: a prime divisor cannot phase-lock with the 16-block macro
# throttle, which is what previously biased the reverb_morph figure.
PROFILE_FLAGS += -DDIAG_PROFILE_DIVISOR=127
ifeq ($(NODMA),1)
BUILDDIR       = build_profile_nodma
else
BUILDDIR       = build_profile
endif
endif

# NODMA applies to both a profile build (compare reverb_T2 between the two) and a
# plain listenable build (A/B the impulses by ear). The flag is set here for both;
# only the non-profile case needs its own build dir, since PROFILE already picked
# build_profile_nodma above.
ifeq ($(NODMA),1)
NODMA_FLAGS = -DDMA2_DISABLED_FOR_DIAGNOSTIC=1
ifneq ($(PROFILE),1)
BUILDDIR    = build_nodma
endif
endif

SOURCES += $(wildcard $(PERIPH)/src/*.c)
SOURCES += $(DEVICE)/src/$(STARTUP)
SOURCES += $(DEVICE)/src/$(SYSTEM)
SOURCES += $(wildcard *.c)

OBJECTS = $(addprefix $(BUILDDIR)/, $(addsuffix .o, $(basename $(SOURCES))))

INCLUDES += -I$(DEVICE)/include \
			-I$(CORE)/include \
			-I$(PERIPH)/include \
			-I\

ELF = $(BUILDDIR)/$(BINARYNAME).elf
HEX = $(BUILDDIR)/$(BINARYNAME).hex
BIN = $(BUILDDIR)/$(BINARYNAME).bin

ARCH = arm-none-eabi
CC = $(ARCH)-gcc
LD = $(ARCH)-ld -v
AS = $(ARCH)-as
OBJCPY = $(ARCH)-objcopy
OBJDMP = $(ARCH)-objdump
GDB = $(ARCH)-gdb

CFLAGS = -g2 -O2 -flto \
          -fno-schedule-insns -fno-schedule-insns2 \
          -fthread-jumps \
          -falign-functions  -falign-jumps \
          -falign-loops  -falign-labels \
          -fcaller-saves \
          -fcrossjumping \
          -fcse-follow-jumps  -fcse-skip-blocks \
          -fdelete-null-pointer-checks \
          -fexpensive-optimizations \
          -fgcse  -fgcse-lm  \
          -findirect-inlining \
          -foptimize-sibling-calls \
          -fpeephole2 \
          -fregmove \
          -freorder-blocks  -freorder-functions \
          -frerun-cse-after-loop  \
          -fsched-interblock  -fsched-spec \
          -fstrict-aliasing -fstrict-overflow \
          -ftree-switch-conversion \
          -ftree-pre \
          -ftree-vrp \
          -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -ftree-vectorize
          
# Causes Freeze on run: -fschedule-insns  -fschedule-insns2 (enabled by -O2, must be disabled)

CFLAGS += -mlittle-endian -mthumb 
CFLAGS +=  -I. -DARM_MATH_CM4 -D'__FPU_PRESENT=1'  $(INCLUDES)  -DUSE_STDPERIPH_DRIVER
CFLAGS += -mcpu=cortex-m4 -mfloat-abi=hard
CFLAGS +=  -mfpu=fpv4-sp-d16 -fsingle-precision-constant -Wdouble-promotion 

# Enable velvet reverb port (comment out to get baseline DLD behaviour)
CFLAGS += -DREVERB_ENABLE
# Bisection knob — drop one notch at a time to find which phase breaks the
# codec ISR on hardware:
#   4 (default) = full reverb        — current corruption
#   3           = no finalize/output  — keeps push/ring/T1/T2 (DMA from SDRAM)
#   2           = no T2 (no DMA)      — only ring_write + T1 from CCM
#   1           = no T1               — only ring_write (CPU SDRAM writes)
#   0           = no phases at all    — only input-buffer fill
CFLAGS += -DREVERB_STAGE=4
# Total tap counts. Block is processed all-at-once from the main loop;
# tune up until diag shows REVERB_BLOCK cycle counts approaching the
# 667 µs / ~112 000-cycle wall-clock budget per block at 48 kHz.
# Three-stage cascade at half codec rate (24 kHz). With N_T0×N_T1×N_T2
# reflection density (~64000 lags for 40×40×40), well past perceptual
# saturation, at half the per-tap cost of the old single-stage 50+25 path.
#   T0 (early)  : window 200 ms — CCM-resident, mono
#   T1 (middle) : window ~666 ms — CCM-resident, mono
#   T2 (late)   : window 8 s   — SDRAM-resident with DMA prefetch, stereo
CFLAGS += -DMAX_T0_TAPS=40
CFLAGS += -DMAX_T1_TAPS=40
# T2 capped at 32 to keep block-time under budget (was 40 → 2% drop rate from
# tail-of-distribution blocks overrunning). Cascade density still 40×40×32 =
# 51200 reflections, well past audible saturation.
CFLAGS += -DMAX_T2_TAPS=32

# Diagnostic FSK firehose. When enabled the right-channel SEND jack
# becomes an FSK-modulated audio stream of cycle-count events. Record
# the send jack at 48 kHz and decode with test/diag_decode.py.
#CFLAGS += -DDIAG_FSK_ENABLE

# Optional-build flags (empty unless PROFILE / NODMA / NOLIM / BISECT is set —
# see the blocks near the top). Appended here because CFLAGS is assigned with
# '=' above, which would discard anything added before that point.
CFLAGS += $(PROFILE_FLAGS) $(NODMA_FLAGS) $(BISECT_FLAGS)

AFLAGS  = -mlittle-endian -mthumb -mcpu=cortex-m4 

LDSCRIPT = $(DEVICE)/$(LOADFILE)
LFLAGS  = -Wl,-Map,$(BUILDDIR)/main.map -T $(LDSCRIPT) -flto -O2 -nostartfiles \
           -mlittle-endian -mthumb -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16

# Math library for velvet reverb (sinf, cosf, expf, sqrtf, powf)
LIBM = $(shell $(CC) $(CFLAGS) -print-file-name=libm.a)
LIBC = $(shell $(CC) $(CFLAGS) -print-file-name=libc.a)
LIBGCC = $(shell $(CC) $(CFLAGS) -print-file-name=libgcc.a)
LIBNOSYS = $(shell $(CC) $(CFLAGS) -print-file-name=libnosys.a)
LIBS = $(LIBM) $(LIBC) $(LIBNOSYS) $(LIBGCC)


all: Makefile $(BIN) $(HEX)

$(BIN): $(ELF)
	$(OBJCPY) -O binary $< $@
	$(OBJDMP) -x --syms $< > $(addsuffix .dmp, $(basename $<))
	ls -l $@ $<

$(HEX): $(ELF)
	$(OBJCPY) --output-target=ihex $< $@

$(ELF): $(OBJECTS) $(wildcard *.h)
	$(CC) $(LFLAGS) -o $@ $(OBJECTS) $(LIBS)


$(BUILDDIR)/%.o: %.c Makefile
	mkdir -p $(dir $@)
	$(CC) -c $(CFLAGS) $< -o $@


$(BUILDDIR)/%.o: %.s Makefile
	mkdir -p $(dir $@)
	$(AS) $(AFLAGS) $< -o $@ > $(addprefix $(BUILDDIR)/, $(addsuffix .lst, $(basename $<)))


flash: $(BIN)
	st-flash write $(BIN) 0x8008000

clean:
	rm -rf build build_profile build_profile_nodma
	
wav: fsk-wav

qpsk-wav: $(BIN)
	PYTHONPATH='.' python3 stm_audio_bootloader/qpsk/encoder.py \
		-t stm32f4 -s 48000 -b 12000 -c 6000 -p 256 \
		$(BIN)

fsk-wav: $(BIN)
	PYTHONPATH='.' python3 stm_audio_bootloader/fsk/encoder.py \
		-s 48000 -b 16 -n 8 -z 4 -p 256 -g 16384 -k 1800 \
		$(BIN)
