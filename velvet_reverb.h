/*
 * velvet_reverb.h — Velvet sparse-noise convolution reverb for DLD
 *
 * Architecture: three-stage cascade (T0 → T1 → T2) at half codec rate
 * (24 kHz), processed from the main loop in 16-sample blocks.
 *
 *   T0: short window (≤ 80 ms), CCM ring, mono — early reflections
 *   T1: medium window (≤ 666 ms), CCM ring, mono — middle scattering
 *   T2: long window (≤ 8 s), SDRAM ring + DMA prefetch, stereo (via
 *       independent L/R read offsets with shared gain magnitude)
 *
 * Per-stage features:
 *   - Tap count fixed at MAX (density is not a control)
 *   - Window / duration (relocation ladder migrates taps as it shrinks)
 *   - Exponential decay envelope on T2 only (T0 taper baked in, T1 flat)
 *   - A pre-delay sustain engine (two modulated, damped feedback delay lines)
 *     sits in front of the cascade and supplies the long tail; the cascade
 *     itself runs purely feedforward. T0 also gets a per-tap Lexicon LFO.
 *
 * Output chain: stereo HPF + LPF biquads at the codec rate; the LPF also
 * doubles as the reconstruction filter for the 2:1 linear-interp upsample.
 *
 * velvet_reverb_push_sample() is called once per CODEC sample from ch0 ISR.
 * velvet_reverb_poll() drains a ready block from the main loop.
 * velvet_reverb_out_left() / out_right() are called from the codec ISRs.
 */

#pragma once
#include <stdint.h>

/* ---- Sample rates ---- */
#define REVERB_FS_HZ   24000

/* ---- SDRAM reverb region ----
 *
 * The T2 ring lives at the very top of SDRAM so it butts up against the end
 * of channel 1's loop region. looping_delay.c shrinks both delay channels
 * by REVERB_SDRAM_RESERVE bytes, which exactly equals the T2 ring's byte
 * count — so channel 1's write head stops at T2_RING_BASE. If T2_RING_BASE
 * sat anywhere lower, channel 1's last REVERB_SDRAM_RESERVE bytes would
 * overlap the T2 ring and the right delay would intermittently play back
 * T2 audio (the prior bug: a 24-kHz reverb tail played at 48 kHz reads as
 * an octave-up "ghost"). */
#define T2_RING_SAMPLES  262144UL
#ifdef VELVET_REVERB_HOST
extern int16_t host_t2_ring_storage[T2_RING_SAMPLES];
#define T2_RING_BASE     ((uintptr_t)host_t2_ring_storage)
#else
/* 0xD2000000 = SDRAM_BASE + SDRAM_SIZE (32 MiB SDRAM ends here).
 * T2 ring occupies [0xD2000000 - T2_RING_SAMPLES*2 .. 0xD2000000). */
#define T2_RING_BASE     (0xD2000000UL - (T2_RING_SAMPLES * 2UL))
#endif
#define T2_RING_MASK     (T2_RING_SAMPLES - 1)

/* T0 window is capped at 80 ms by the macro (≈1920 samples @ 24 kHz); the
 * deepest read is window + tap-LFO swing (~120) ≈ 2040 back. 4096 (171 ms)
 * gives >2x margin and frees 8 KB CCM vs the old 8192. */
#define T0_RING_SAMPLES  4096UL
#define T0_RING_MASK     (T0_RING_SAMPLES - 1)

#define T1_RING_SAMPLES  16384UL
#define T1_RING_MASK     (T1_RING_SAMPLES - 1)

/* ---- Pre-delay sustain-engine lines ----
 * Two modulated, damped feedback delay lines sit IN FRONT of the velvet
 * cascade (replacing the old per-stage/global recirculation). They store
 * float samples (normalised ±1, node-clamped ±4) so the high-feedback loop
 * doesn't accumulate int16 quantisation noise. Placed in SDRAM just below
 * the T2 ring (reserved together via REVERB_SDRAM_RESERVE), since 2×64 KB
 * doesn't fit the ~9 KB of free CCM. Line length is a power of two so the
 * read/write wraps are a cheap mask; it covers PRE_DELAY_MAX + 2×PRE_MOD. */
#define PRE_DELAY_LINE_SAMPLES  16384UL
#define PRE_DELAY_LINE_MASK     (PRE_DELAY_LINE_SAMPLES - 1)
#define PRE_DELAY_MAX_SAMPLES   12000   /* 0.5 s @ 24 kHz (firmware cap; JS = 1 s) */
#define PRE_MOD_MAX_SAMPLES     720     /* 0.030 s @ 24 kHz (depth = 1 swing) */

#ifdef VELVET_REVERB_HOST
extern float host_predelay_a_storage[PRE_DELAY_LINE_SAMPLES];
extern float host_predelay_b_storage[PRE_DELAY_LINE_SAMPLES];
#define PRE_DELAY_A_BASE  ((uintptr_t)host_predelay_a_storage)
#define PRE_DELAY_B_BASE  ((uintptr_t)host_predelay_b_storage)
#else
/* Two float lines directly below the T2 ring: A then B, contiguous. */
#define PRE_DELAY_A_BASE  (T2_RING_BASE - 2UL * PRE_DELAY_LINE_SAMPLES * 4UL)
#define PRE_DELAY_B_BASE  (T2_RING_BASE - 1UL * PRE_DELAY_LINE_SAMPLES * 4UL)
#endif

/* Reserve = T2 ring (int16) + both pre-delay lines (float), so channel 1's
 * write head stops at PRE_DELAY_A_BASE (below the whole reverb region). */
#define REVERB_SDRAM_RESERVE  (T2_RING_SAMPLES * 2UL + 2UL * PRE_DELAY_LINE_SAMPLES * 4UL)

/* ---- Block geometry ---- */
#define REVERB_BLOCK         16
#define REVERB_OUT_BLOCK     (REVERB_BLOCK * 2)

/* Depth of the ISR-to-main-loop input queue, in blocks. (REVERB_IN_NBUF - 1)
 * blocks is how late the main loop may be before an input block is dropped; see
 * the rationale at in_ring in velvet_reverb.c. Here rather than in the .c so the
 * host harness can report the tolerance it is measuring. */
#ifndef REVERB_IN_NBUF
#define REVERB_IN_NBUF       16
#endif

/* ---- Tap limits (compile-time upper bounds; runtime target counts can go below) ---- */
#ifndef MAX_T0_TAPS
#define MAX_T0_TAPS  40
#endif
#ifndef MAX_T1_TAPS
#define MAX_T1_TAPS  40
#endif
#ifndef MAX_T2_TAPS
#define MAX_T2_TAPS  32
#endif

/* ---- Maximum windows (samples at fs_reverb, also user-tunable upper bounds) ---- */
#define T0_WINDOW_MAX_SAMPLES   4800    /* 200 ms @ 24 kHz */
#define T1_WINDOW_MAX_SAMPLES   16000   /* ~666 ms @ 24 kHz */
#define T2_DURATION_MAX_SAMPLES 192000  /* 8.0 s   @ 24 kHz */

#ifndef TAP_GAIN_HEADROOM
#define TAP_GAIN_HEADROOM  1.0f
#endif

/* ---- Tap-ladder geometry ----
 * Each tap has a chain of candidate read positions descending geometrically
 * from its main offset. As the window shrinks past each rung, the tap
 * migrates one rung down via a bell-shaped window-gain that hits 0 at every
 * rung boundary — the discrete position swap happens in silence (no click).
 * MAX_LADDER_LEVELS covers T2's 8 s span comfortably with a 0.25 ratio. */
#define MAX_LADDER_LEVELS  8
#define LADDER_RATIO_Q15   8192   /* 0.25 in Q1.15 — used in the ladder builder */

/* ---- Tap arrays (always generated at MAX count + MAX window; runtime
 *      morphing decides how many of them are active and how far the
 *      window extends.) ---- */
/* Density is fixed at MAX (all taps always active — there is no Density
 * control), so the per-tap removal-rank arrays are gone. Each stage always
 * generates MAX taps once at init; the relocation ladder still migrates them
 * one-at-a-time as the Decay-driven window shrinks. */
extern uint32_t t0TapOffsets[MAX_T0_TAPS];
extern int16_t  t0TapGains [MAX_T0_TAPS];   /* base gain incl. taper (no envelope, no comp) */

extern uint32_t t1TapOffsets[MAX_T1_TAPS];
extern int16_t  t1TapGains [MAX_T1_TAPS];

extern uint32_t t2TapOffsetsL[MAX_T2_TAPS];
extern uint32_t t2TapOffsetsR[MAX_T2_TAPS];
extern int16_t  t2TapGains   [MAX_T2_TAPS];

/* ---- Mutable runtime globals ----
 *
 * Targets are smoothed per-block with a one-pole IIR. Defaults match the
 * Phase-1 maxed-out state (full tap count, full window, near-linear decay,
 * recirc near clamp). Poke these from a debugger or hook them up to
 * hardware controls (Phase 3). */

/* Reverb send level (right MIX pot, equal-power LUT in params.c). Scales
 * the audio that's fed INTO the reverb via velvet_reverb_push_sample.
 * The reverb's stereo output is always mixed unscaled into the final
 * output — `reverb_send` only controls how hard the reverb is driven, not
 * how much of its output you hear. */
extern float reverb_send;

/* Dry-path gain applied to the delay/wet mix BEFORE the reverb tail is
 * summed in. Set in params.c so that the top 10% of the right MIX pot
 * fades this to 0, leaving 100% wet reverb at the knob's max. */
extern float reverb_dry_gain;

/* Output filter cutoffs (codec rate, applied post-upsample) */
extern float reverb_lpf_hz;
extern float reverb_hpf_hz;

/* Per-stage window/duration target (driven by the Decay macro) */
extern float reverb_t0_window_ms_target;     /* 0..~200 */
extern float reverb_t1_window_ms_target;     /* 0..~666 */
extern float reverb_t2_duration_s_target;    /* 0..~8 */

/* ---- Pre-delay sustain engine (replaces recirculation) ----
 * Two feedback delay lines A/B with one shared loop gain, shared low/high
 * shelves (tone + damping), per-line modulation, ~12 Hz DC block, input
 * ducking, and a dry/wet crossfade into the cascade. JS-matched defaults. */
extern float reverb_feedback;            /* shared loop gain (clamped 0.999) */
extern float reverb_predelay_a_s;        /* line A delay time, seconds */
extern float reverb_predelay_b_s;        /* line B delay time, seconds */
extern float reverb_delay_mix;           /* 0 = dry input, 1 = full pre-delay output */
extern float reverb_fb_low_shelf_hz;
extern float reverb_fb_low_shelf_db;
extern float reverb_fb_high_shelf_hz;
extern float reverb_fb_high_shelf_db;
extern float reverb_fb_mod_depth;        /* 0..1 (× PRE_MOD_MAX_SAMPLES) */
extern float reverb_fb_mod_rate;         /* Hz */
extern float reverb_duck_amount;         /* 0..1 max feedback reduction */
extern float reverb_duck_release_loops;  /* loop traversals, not seconds */

/* T0 (early-reflection) per-tap Lexicon LFO — golden-angle decorrelated. */
extern float reverb_t0_tap_mod_depth;    /* 0..1 (× ~5 ms swing) */
extern float reverb_t0_tap_mod_rate;     /* Hz */

/* ---- API ---- */
void velvet_reverb_init(void);
void velvet_reverb_regenerate_taps(void);

void velvet_reverb_push_sample(int16_t input);
int16_t velvet_reverb_out_left(void);
int16_t velvet_reverb_out_right(void);

void velvet_reverb_poll(void);

/* Count of input blocks discarded because poll had not consumed the previous
 * one — i.e. the main loop was late by more than one block period. Each drop
 * splices REVERB_BLOCK samples out of the reverb's input and is audible as a
 * whoosh, so this is the objective measure of how much main-loop lateness the
 * engine tolerates. Exposed for the host harness. */
extern volatile uint32_t input_drop_count;
/* Peak input-queue occupancy in blocks, out of (REVERB_IN_NBUF - 1) usable. */
extern volatile uint8_t input_queue_max;

/* ---- Macro system ----
 *
 * Two macros — Decay and Tone — each take a 0..1 pot position. Every mapped
 * param uses the full range: u lerps bound_min..bound_max (exp where both > 0).
 * Decay is driven by the right REGEN pot; Tone by the right LEVEL pot. */
void velvet_reverb_apply_decay_macro  (float v);
void velvet_reverb_apply_tone_macro   (float v);
void velvet_reverb_recompute_macros   (void);
