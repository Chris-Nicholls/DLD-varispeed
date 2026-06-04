BINARYNAME = main

STARTUP = startup_stm32f427_437xx.s
SYSTEM = system_stm32f4xx.c
LOADFILE = stm32f427.ld

DEVICE = stm32/device
CORE = stm32/core
PERIPH = stm32/periph

BUILDDIR = build

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
AFLAGS  = -mlittle-endian -mthumb -mcpu=cortex-m4 

LDSCRIPT = $(DEVICE)/$(LOADFILE)
LFLAGS  = -Wl,-Map,main.map -T $(LDSCRIPT) -flto -O2 -nostartfiles \
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


$(BUILDDIR)/%.o: %.c
	mkdir -p $(dir $@)
	$(CC) -c $(CFLAGS) $< -o $@


$(BUILDDIR)/%.o: %.s
	mkdir -p $(dir $@)
	$(AS) $(AFLAGS) $< -o $@ > $(addprefix $(BUILDDIR)/, $(addsuffix .lst, $(basename $<)))


flash: $(BIN)
	st-flash write $(BIN) 0x8008000

clean:
	rm -rf build
	
wav: fsk-wav

qpsk-wav: $(BIN)
	PYTHONPATH='.' python3 stm_audio_bootloader/qpsk/encoder.py \
		-t stm32f4 -s 48000 -b 12000 -c 6000 -p 256 \
		$(BIN)

fsk-wav: $(BIN)
	PYTHONPATH='.' python3 stm_audio_bootloader/fsk/encoder.py \
		-s 48000 -b 16 -n 8 -z 4 -p 256 -g 16384 -k 1800 \
		$(BIN)
