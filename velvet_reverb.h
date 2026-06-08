/*
 * velvet_reverb.h — Velvet sparse-noise convolution reverb for DLD
 *
 * Architecture: three-stage cascade (T0 → T1 → T2) at half codec rate
 * (24 kHz), processed from the main loop in 16-sample blocks.
 *
 *   T0: short window (≤ 200 ms), CCM ring, mono — early reflections
 *   T1: medium window (≤ 666 ms), CCM ring, mono — middle scattering
 *   T2: long window (≤ 8 s), SDRAM ring + DMA prefetch, stereo (via
 *       independent L/R read offsets with shared gain magnitude)
 *
 * Per-stage features (all dynamically tunable at runtime via the
 * reverb_*_target_* globals — see below):
 *   - Tap count (with smooth density morph via tap-survival rank)
 *   - Window / duration (with smooth fade at the moving edge)
 *   - Decay envelope (exp↔linear blend) for T1 and T2
 *   - Recirculation amount: 2 taps on T0/T1, 3 taps on T2, with LFO
 *     modulation of read positions (anti-comb) and one-pole damping LPF
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

#define T0_RING_SAMPLES  8192UL
#define T0_RING_MASK     (T0_RING_SAMPLES - 1)

#define T1_RING_SAMPLES  16384UL
#define T1_RING_MASK     (T1_RING_SAMPLES - 1)

#define REVERB_SDRAM_RESERVE  (T2_RING_SAMPLES * 2)

/* ---- Block geometry ---- */
#define REVERB_BLOCK         16
#define REVERB_OUT_BLOCK     (REVERB_BLOCK * 2)

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
extern uint32_t t0TapOffsets[MAX_T0_TAPS];
extern int16_t  t0TapGains [MAX_T0_TAPS];   /* base gain incl. taper (no envelope, no comp) */
extern float    t0TapRanks [MAX_T0_TAPS];   /* removal-order rank (lower = kept longer) */

extern uint32_t t1TapOffsets[MAX_T1_TAPS];
extern int16_t  t1TapGains [MAX_T1_TAPS];
extern float    t1TapRanks [MAX_T1_TAPS];

extern uint32_t t2TapOffsetsL[MAX_T2_TAPS];
extern uint32_t t2TapOffsetsR[MAX_T2_TAPS];
extern int16_t  t2TapGains   [MAX_T2_TAPS];
extern float    t2TapRanks   [MAX_T2_TAPS];

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

/* Per-stage recirculation amount (clamped to 0.85 in DSP for stability) */
extern float reverb_t0_recirc;
extern float reverb_t1_recirc;
extern float reverb_t2_recirc;

/* Global recirculation amount: T2 tail → T0 input feedback loop (0..clamp). */
extern float reverb_global_recirc;

/* Per-stage active-tap count target (0..MAX_T{N}_TAPS, fractional allowed) */
extern float reverb_t0_count_target;
extern float reverb_t1_count_target;
extern float reverb_t2_count_target;

/* Per-stage window/duration target */
extern float reverb_t0_window_ms_target;     /* 0..~200 */
extern float reverb_t1_window_ms_target;     /* 0..~666 */
extern float reverb_t2_duration_s_target;    /* 0..~8 */

/* Per-stage decay shape (0 = exponential, 1 = linear). T0 has no envelope. */
extern float reverb_t1_decay_shape;
extern float reverb_t2_decay_shape;

/* Recirc mod depth as percent of each stage's window (~0..5 typical). */
extern float reverb_recirc_mod_depth_pct;

/* Feedback high-shelf damping: 0 = full LP damping, 1 = none (HF decay ratio). */
extern float reverb_recirc_damp_hf_gain;

/* ---- API ---- */
void velvet_reverb_init(void);
void velvet_reverb_regenerate_taps(void);

void velvet_reverb_push_sample(int16_t input);
int16_t velvet_reverb_out_left(void);
int16_t velvet_reverb_out_right(void);

void velvet_reverb_poll(void);

/* ---- Macro system ----
 *
 * Three macros — Density, Decay, Tone — each take a 0..1 position. Each
 * macro has a fixed list of per-param mappings (with lo/hi sub-range, both
 * in [0,1]). When multiple macros map the same param, their effective-t's
 * (per-mapping lerp(lo, hi, u)) are multiplied together; the product lerps
 * the param's bounds.
 *
 * Bounds + mappings mirror the JS prototype's velvet_param_bounds /
 * velvet_macros snapshot; baked-in here. Density is driven by the right
 * channel's LEVEL pot, Decay by the right REGEN pot (see params.c). Tone
 * has no hardware knob — set via velvet_reverb_apply_tone_macro(). */
void velvet_reverb_apply_density_macro(float v);
void velvet_reverb_apply_decay_macro  (float v);
void velvet_reverb_apply_tone_macro   (float v);
void velvet_reverb_recompute_macros   (void);
