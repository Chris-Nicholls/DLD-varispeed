/*
 * velvet_reverb.c — Velvet sparse-noise convolution reverb for DLD
 *
 * Three-stage cascade (T0 → T1 → T2) at half codec rate (24 kHz), processed
 * from the main loop in 16-sample blocks (= 32 codec samples, ~667 µs).
 *
 *   codec → push (avg 2:1 decimate) → T0 ring (CCM)
 *           T0 sparse conv  +  T0 recirc (2 modulated, damped taps)
 *             ↓ saturate
 *           T1 ring (CCM)
 *           T1 sparse conv  +  T1 recirc (2 modulated, damped taps)
 *             ↓ saturate
 *           T2 ring (SDRAM, DMA-prefetched)
 *           T2 sparse conv (stereo via independent L/R offsets, shared gain)
 *             +  T2 recirc (3 modulated, damped taps, mono)
 *             ↓ saturate + 2× linear-interp upsample
 *           HPF + LPF biquads @ 48 kHz (LPF doubles as reconstruction filter)
 *             ↓
 *           codec output
 *
 * Dynamic-parameter infrastructure (Phase 2):
 *
 *   - Tap counts, window sizes, decay shapes, and recirc amounts are all
 *     mutable globals smoothed per-block with a one-pole IIR.
 *   - Taps are generated once at IR-gen time at MAX count + MAX window with
 *     just sign + 1/√N (no envelope, no taper for T1/T2). T0 keeps cosine
 *     tail-taper. Each tap also gets a "removal rank" computed at IR-gen
 *     based on neighbour spacing (taps with closest neighbours fade out
 *     first as density drops).
 *   - Each block we recompute the *effective* per-tap gain (int16 Q15):
 *         base × density-fade × window-fade × envelope × gain-comp
 *     and the convolution inner loops read from these effGains arrays.
 *     This lets density, window, envelope, and loudness compensation all
 *     change smoothly without re-running tap generation.
 *   - The effGains recompute is round-robin across the three stages (one
 *     per block) to keep the per-block update cost flat.
 *
 * KEY INVARIANT: input_buf is ALWAYS written linearly 0..15, ensuring
 * chronological order when committed to the T0 ring.
 */

#ifdef VELVET_REVERB_HOST
#include "host_shim.h"
#else
#include "stm32f4xx.h"
#endif
#include "velvet_reverb.h"
#include <math.h>
#include <string.h>

#ifndef VELVET_REVERB_HOST
#include "diag_log.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TWO_PI_F  6.28318530717958647692f
#define PI_F      3.14159265358979323846f

#ifdef VELVET_REVERB_HOST
#include <limits.h>
__attribute__((always_inline)) static inline int32_t qadd_sat(int32_t a, int32_t b)
{
    int64_t r = (int64_t)a + (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}
__attribute__((always_inline)) static inline int32_t smulbb(uint32_t a, uint32_t b)
{
    return (int32_t)(int16_t)(a & 0xFFFFu) * (int32_t)(int16_t)(b & 0xFFFFu);
}
__attribute__((always_inline)) static inline int32_t smultb(uint32_t a, uint32_t b)
{
    return (int32_t)(int16_t)(a >> 16) * (int32_t)(int16_t)(b & 0xFFFFu);
}
#else
__attribute__((always_inline)) static inline int32_t qadd_sat(int32_t a, int32_t b)
{
    int32_t r;
    __asm volatile ("qadd %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__attribute__((always_inline)) static inline int32_t smulbb(uint32_t a, uint32_t b)
{
    int32_t r;
    __asm volatile ("smulbb %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__attribute__((always_inline)) static inline int32_t smultb(uint32_t a, uint32_t b)
{
    int32_t r;
    __asm volatile ("smultb %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
#endif

/* ==== CCM placement ==== */
#ifdef VELVET_REVERB_HOST
#define CCM_ATTR __attribute__((aligned(4)))
#else
#define CCM_ATTR __attribute__((aligned(4), section(".ccmdata")))
#endif

/* May-alias typedef for fused 32-bit reads/writes on int16_t buffers.
 * Without this, casting (uint32_t *)<int16_t *> violates strict aliasing —
 * GCC may reorder the read/write w.r.t. int16_t accesses to the same memory,
 * producing intermittent wrong-data bugs. With __attribute__((may_alias)),
 * GCC knows this type can alias any other and won't optimize across it. */
typedef uint32_t u32_alias __attribute__((may_alias));

/* ==== Rings ==== */
static int16_t t0_ring[T0_RING_SAMPLES] CCM_ATTR;
static int16_t t1_ring[T1_RING_SAMPLES] CCM_ATTR;
static int16_t * const t2_ring = (int16_t *)T2_RING_BASE;
static uint32_t ring_write_idx = 0;

/* ==== Tap arrays (generated once at max count + max window) ==== */
uint32_t t0TapOffsets[MAX_T0_TAPS];
int16_t  t0TapGains [MAX_T0_TAPS];
float    t0TapRanks [MAX_T0_TAPS];
int      t0TapCount = 0;

uint32_t t1TapOffsets[MAX_T1_TAPS];
int16_t  t1TapGains [MAX_T1_TAPS];
float    t1TapRanks [MAX_T1_TAPS];
int      t1TapCount = 0;

uint32_t t2TapOffsetsL[MAX_T2_TAPS];
uint32_t t2TapOffsetsR[MAX_T2_TAPS];
int16_t  t2TapGains   [MAX_T2_TAPS];
float    t2TapRanks   [MAX_T2_TAPS];
int      t2TapCount = 0;

/* ==== Per-tap position ladders ====
 * Each tap has up to MAX_LADDER_LEVELS candidate read positions, generated
 * at IR-gen time. Rung 0 is the main offset; subsequent rungs are
 * (previous × LADDER_RATIO) ± per-rung jitter. As the window shrinks the
 * tap migrates one rung down at a time; the swap happens at gain = 0 so
 * it's inaudible (see compute_tap_states below).
 *
 * T2 has independent L/R ladders sharing one per-rung jitter draw — this
 * is what keeps L/R offset divergence bounded across rungs (without
 * shared jitter, deep-rung divergence grows into a ping-pong delay). */
static uint16_t t0TapLadder [MAX_T0_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t0TapLadderCount[MAX_T0_TAPS]                 CCM_ATTR;
static uint16_t t1TapLadder [MAX_T1_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t1TapLadderCount[MAX_T1_TAPS]                 CCM_ATTR;
static uint16_t t2TapLadderL[MAX_T2_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint16_t t2TapLadderR[MAX_T2_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t2TapLadderCount[MAX_T2_TAPS]                 CCM_ATTR;

/* ==== Per-block effective read offsets ====
 * Updated each block by compute_tap_states based on the current window —
 * each tap is at whichever ladder rung's band contains the window. The
 * convolution inner loops read from these instead of from tNTapOffsets. */
static uint16_t effOffT0 [MAX_T0_TAPS] CCM_ATTR;
static uint16_t effOffT1 [MAX_T1_TAPS] CCM_ATTR;
static uint16_t effOffT2L[MAX_T2_TAPS] CCM_ATTR;
static uint16_t effOffT2R[MAX_T2_TAPS] CCM_ATTR;

/* ==== Effective per-tap gains (rebuilt per block, round-robin across stages) ==== */
static int16_t effGains_t0[MAX_T0_TAPS] CCM_ATTR;
static int16_t effGains_t1[MAX_T1_TAPS] CCM_ATTR;
static int16_t effGains_t2[MAX_T2_TAPS] CCM_ATTR;

/* The old precomputed 2D energy LUT and per-tap env LUT have been replaced
 * by per-block ladder-based recompute (see compute_tap_states + recompute
 * functions below). The ladder makes density approximately constant across
 * the window range, so the per-stage energy is dominated by the per-tap
 * effective window-gain — fast to sum on the fly, no 2D table required. */

/* ==== Mutable globals — driven by the macro system (see below). Initial
 *      values are what the macros produce at their default (D=0, Dec=0,
 *      Tone=0.83) positions; will be overwritten on the first macro call. */
float reverb_send      = 0.7f;   /* updated each params iter from right MIX pot */
float reverb_dry_gain  = 1.0f;   /* fades to 0 over top 10% of right MIX pot */
float reverb_lpf_hz    = 4500.0f;
float reverb_hpf_hz    = 100.0f;
float reverb_t0_recirc = 0.0f;
float reverb_t1_recirc = 0.25f;
float reverb_t2_recirc = 0.0f;

/* Global recirculation: feeds a few modulated taps of the long T2 tail back
 * into T0's input, forming a T0→T1→T2→T0 loop on top of the per-stage recirc.
 * Greatly extends/thickens the tail. Clamped to RECIRC_AMOUNT_CLAMP in the DSP
 * for stability. 0 = off (default). */
float reverb_global_recirc = 0.0f;

float reverb_t0_count_target     = 1.0f;
float reverb_t1_count_target     = 0.0f;
float reverb_t2_count_target     = 10.0f;

float reverb_t0_window_ms_target  = 40.0f;
float reverb_t1_window_ms_target  = 100.0f;
float reverb_t2_duration_s_target = 0.213f;

float reverb_t1_decay_shape = 0.0f;
float reverb_t2_decay_shape = 0.0f;
float reverb_recirc_mod_depth_pct = 0.06f;

/* Feedback high-shelf damping amount applied in every recirc loop, between the
 * one-pole LP damping and the 100 Hz HPF. 0 = pure LP (darkest, HF dies fast);
 * 1 = no HF damping (flat); in between, HF decays faster than LF by this ratio. */
float reverb_recirc_damp_hf_gain = 0.1f;

/* ==== Hardcoded constants (recirc modulation + damping, gain-comp clamp) ====
 * Rates / damping match the JS app's velvet_buffers_sliders snapshot
 * (paramRecircModRate=0.55, paramRecircDamp=10800). Not macro-driven. */
#define RECIRC_MOD_RATE_HZ    0.5f
#define RECIRC_DAMP_FREQ_HZ   2600.0f
#define RECIRC_HPF_FREQ_HZ    100.0f   /* one-pole HPF in each feedback loop — blocks
                                        * sub-100 Hz / DC accumulation in the recirc */
#define RECIRC_AMOUNT_CLAMP   1.0f
#define TAP_COMP_CLAMP        2.0f       /* max gain-comp boost (avoid int16 overflow) */

/* Absolute floor (in samples) on the recirc position-modulation depth.
 * The per-stage mod depth is normally a fraction of each stage's window, so at
 * low decay (short windows) it collapses toward zero — and the recirc tap
 * ladders bottom out near TAP_MIN_OFFSET, forming ~130-280 sample feedback
 * loops. An un-modulated loop that short rings as a fixed low-mid comb
 * (~110-190 Hz). This floor keeps a minimum continuous modulation so those
 * short loops stay smeared instead of resonating. Only affects output when a
 * stage's recirc is active (the recirc fns early-return at amt<=0). Set to 0
 * to restore the pure window-proportional behaviour. */
#define RECIRC_MOD_DEPTH_MIN_SAMPLES  6.0f

/* Minimum read offset (in samples) for the *recirc* tap ladders — a recirc-only
 * floor, larger than the main taps' TAP_MIN_OFFSET. Stops the recirc ladders
 * descending into ~130-280 sample rungs whose feedback loops ring as a low-mid
 * comb (~110-190 Hz). At 480 the shortest recirc loop's comb fundamental is
 * 24000/480 = 50 Hz (sub-bass, below the output HPF). When the morphed window
 * shrinks below the deepest remaining rung, compute_tap_states fades the recirc
 * tap out (gain 0) rather than forming a short loop. Main taps are unaffected. */
#define RECIRC_MIN_OFFSET  480

/* Mid-side stereo width applied to the final T2 output in do_finalize.
 * 1.0 = neutral, <1 narrows toward mono, >1 widens by boosting the
 * decorrelated (L-R) component. Only T2 is stereo (T0/T1 are mono), so this
 * directly scales that decorrelation. Hard-coded for now. */
#define REVERB_STEREO_WIDTH   1.79f

/* Internal headroom. The whole int16 cascade runs at REVERB_HEADROOM × level
 * (cooler), and the output is scaled back up by REVERB_OUT_MAKEUP = 1/headroom.
 * This mirrors the JS float path's headroom: inter-stage values that would push
 * past int16 full-scale (conv-sum peaks, recirc feedback, etc.) now sit below
 * it, so the soft knee (below) rarely engages — matching the JS until a true
 * over. Cost: the internal computation runs ~6 dB closer to the int16 LSB at
 * 0.5, so the quietest tail loses ~1 bit of SNR. Raise toward 1.0 if the decay
 * sounds grainy; lower toward 0.5 if clipping persists. The pre-T2 saturator is
 * gain-compensated (see apply_pre_t2_sat) so its drive/character is unchanged. */
#define REVERB_HEADROOM    0.5f
#define REVERB_OUT_MAKEUP  (1.0f / REVERB_HEADROOM)

/* Soft-knee threshold (full-scale units) for the inter-stage / output
 * saturators. |x| <= threshold passes through linearly (transparent, like the
 * JS); above it a C1 rational knee rounds off toward ±1.0 instead of the old
 * hard int16 clamp. With the headroom above, signals normally stay under the
 * threshold; the knee only shapes genuine overs. */
#define SOFT_KNEE_T  0.80f

#define T0_RECIRC_TAPS    2
#define T1_RECIRC_TAPS    2
#define T2_RECIRC_TAPS    3
#define GLOBAL_RECIRC_TAPS 3   /* T2 tail -> T0 input global feedback taps */
#define RECIRC_MOD_TOTAL  (T0_RECIRC_TAPS + T1_RECIRC_TAPS + T2_RECIRC_TAPS + GLOBAL_RECIRC_TAPS)

/* ==== Macro system ==== Mirrors the JS prototype's velvet_param_bounds +
 * velvet_macros snapshot. Three macros (Density, Decay, Tone), each with a
 * 0..1 slider value, fixed mappings to a subset of params. Multiplicative
 * combination across macros — see velvet_reverb_recompute_macros below. */
typedef enum {
    MP_T0_COUNT, MP_T0_WINDOW, MP_T0_RECIRC,
    MP_T1_COUNT, MP_T1_WINDOW, MP_T1_DECAY, MP_T1_RECIRC,
    MP_T2_COUNT, MP_T2_DURATION, MP_T2_DECAY, MP_T2_RECIRC,
    MP_GLOBAL_RECIRC,
    MP_LPF, MP_HPF,
    MP_COUNT
} macro_param_id_t;

typedef struct {
    float *target;
    float bound_min;
    float bound_max;
    uint8_t use_exp;       /* 1 = exp lerp when bounds both > 0; else linear */
} macro_param_t;

typedef struct {
    macro_param_id_t param_id;
    float lo, hi;
} macro_mapping_t;

typedef struct {
    float value;
    float curve_power;      /* u' = u^curve_power applied before mapping; 1.0 = linear */
    int   num_mappings;
    const macro_mapping_t *mappings;
} macro_t;

/* Bounds + use_exp mirror the JS velvet_param_bounds + recomputeAllMacroParams:
 * exp lerp where the last macro mapping the param qualifies (decay/tone → all;
 * density → tap-count only) AND min>0, max>0, min!=max — which resolves to exp
 * for T0 count, T0/T1 window, T2 duration, LPF, HPF; linear for the rest.
 * Stage remap JS→firmware: paramT1→T0, paramTm→T1, paramT2Taps→T2. */
static macro_param_t reverb_macro_params[MP_COUNT] = {
    [MP_T0_COUNT]     = { &reverb_t0_count_target,      1.0f,   40.0f,    1 }, /* paramT1 */
    [MP_T0_WINDOW]    = { &reverb_t0_window_ms_target,  80.0f,  100.0f,   1 }, /* paramT1Window */
    [MP_T0_RECIRC]    = { &reverb_t0_recirc,            0.0f,   0.0f,     0 }, /* paramT1Recirc */
    [MP_T1_COUNT]     = { &reverb_t1_count_target,      0.0f,   40.0f,    0 }, /* paramTm */
    [MP_T1_WINDOW]    = { &reverb_t1_window_ms_target,  150.0f, 660.0f,   1 }, /* paramTmWindow */
    [MP_T1_DECAY]     = { &reverb_t1_decay_shape,       0.0f,   2.0f,     0 }, /* paramTmDecay */
    [MP_T1_RECIRC]    = { &reverb_t1_recirc,            0.0f,   0.0f,     0 }, /* paramTmRecirc */
    [MP_T2_COUNT]     = { &reverb_t2_count_target,      0.0f,   32.0f,    0 }, /* paramT2Taps */
    [MP_T2_DURATION]  = { &reverb_t2_duration_s_target, 0.1f,   3.0f,     1 }, /* paramT2Duration */
    [MP_T2_DECAY]     = { &reverb_t2_decay_shape,       0.0f,   1.0f,     0 }, /* toggleT2Linear */
    [MP_T2_RECIRC]    = { &reverb_t2_recirc,            0.0f,   1.0f,     0 }, /* paramT2Recirc */
    [MP_GLOBAL_RECIRC]= { &reverb_global_recirc,        0.0f,   0.2f,     0 }, /* paramGlobalRecirc */
    [MP_LPF]          = { &reverb_lpf_hz,               400.0f, 10000.0f, 1 }, /* paramLpf */
    [MP_HPF]          = { &reverb_hpf_hz,               20.0f,  300.0f,   1 }, /* paramHpf */
};

static const macro_mapping_t density_mappings[] = {
    { MP_T0_COUNT,      0.0f, 1.0f },   /* paramT1 */
    { MP_T2_COUNT,      0.0f, 1.0f },   /* paramT2Taps */
    { MP_T1_COUNT,      0.0f, 1.0f },   /* paramTm */
    { MP_GLOBAL_RECIRC, 0.0f, 1.0f },   /* paramGlobalRecirc */
};
static const macro_mapping_t decay_mappings[] = {
    { MP_T2_DURATION,   0.0f, 1.0f },   /* paramT2Duration */
    { MP_T0_WINDOW,     0.0f, 1.0f },   /* paramT1Window */
    { MP_T1_WINDOW,     0.0f, 1.0f },   /* paramTmWindow */
    { MP_T1_DECAY,      0.0f, 1.0f },   /* paramTmDecay */
    { MP_T2_DECAY,      0.0f, 1.0f },   /* toggleT2Linear */
    { MP_T2_RECIRC,     0.0f, 0.6f },   /* paramT2Recirc (hi 0.6) */
    { MP_T1_RECIRC,     0.0f, 1.0f },   /* paramTmRecirc */
    { MP_GLOBAL_RECIRC, 0.0f, 1.0f },   /* paramGlobalRecirc */
    { MP_T2_COUNT,      0.0f, 1.0f },   /* paramT2Taps */
    { MP_T0_COUNT,      0.0f, 1.0f },   /* paramT1 */
};
static const macro_mapping_t tone_mappings[] = {
    { MP_LPF, 0.0f, 1.0f },
    { MP_HPF, 0.0f, 1.0f },
};

#define M_COUNT(arr) (int)(sizeof(arr) / sizeof((arr)[0]))
/* No pre-curve (curve_power = 1.0) — the JS recomputeAllMacroParams uses the
 * slider value directly. Density/Decay values are overwritten each iteration
 * from the LEVEL/REGEN pots (see params.c); Tone has no pot, so its default
 * (0.62) is the value the user dialled in. */
static macro_t reverb_macros[3] = {
    { 0.0f,  1.0f, M_COUNT(density_mappings), density_mappings },
    { 0.0f,  1.0f, M_COUNT(decay_mappings),   decay_mappings   },
    { 0.62f, 1.0f, M_COUNT(tone_mappings),    tone_mappings    },
};

/* Macro recompute is ~14 powf calls (~17 µs). Called from params.c at
 * main-loop rate (~1 kHz), but the morph IIRs downstream have a 30 ms tau,
 * so anything faster than ~100 Hz is wasted work. apply_*_macro now just
 * stores the value + sets a dirty flag; the actual recompute happens at
 * audio-block rate, throttled. */
static volatile uint8_t macros_dirty = 1;   /* force initial recompute */

void velvet_reverb_recompute_macros(void)
{
    /* tByParam < 0 sentinel = no macro maps this param. */
    float tByParam[MP_COUNT];
    for (int p = 0; p < MP_COUNT; p++) tByParam[p] = -1.0f;
    for (int m = 0; m < 3; m++) {
        float u = reverb_macros[m].value;
        if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
        /* Pre-curve on macro position. powf(u, 3) is special-cased to a
         * triple multiply to avoid the ~200-cycle generic powf for the
         * common cubic case. */
        float cp = reverb_macros[m].curve_power;
        if (cp == 3.0f) {
            u = u * u * u;
        } else if (cp != 1.0f) {
            u = powf(u, cp);
        }
        const macro_mapping_t *maps = reverb_macros[m].mappings;
        int n = reverb_macros[m].num_mappings;
        for (int i = 0; i < n; i++) {
            float tMap = maps[i].lo + u * (maps[i].hi - maps[i].lo);
            int pid = maps[i].param_id;
            if (tByParam[pid] < 0.0f) tByParam[pid] = 1.0f;
            tByParam[pid] *= tMap;
        }
    }
    for (int p = 0; p < MP_COUNT; p++) {
        if (tByParam[p] < 0.0f) continue;
        const macro_param_t *mp = &reverb_macro_params[p];
        float t = tByParam[p];
        float val;
        if (mp->use_exp && mp->bound_min > 0.0f && mp->bound_max > 0.0f && mp->bound_min != mp->bound_max) {
            val = mp->bound_min * powf(mp->bound_max / mp->bound_min, t);
        } else {
            val = mp->bound_min + t * (mp->bound_max - mp->bound_min);
        }
        *mp->target = val;
    }
}

/* Dead-band of ~1/4096 — comfortably above 12-bit ADC LSB noise but below
 * any audible step (cosumes a single ADC count of movement). */
#define MACRO_VALUE_DEADBAND  (1.0f / 4096.0f)

static inline void set_macro_value(int idx, float v) {
    if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
    if (fabsf(v - reverb_macros[idx].value) < MACRO_VALUE_DEADBAND) return;
    reverb_macros[idx].value = v;
    macros_dirty = 1;
}

void velvet_reverb_apply_density_macro(float v) { set_macro_value(0, v); }
void velvet_reverb_apply_decay_macro  (float v) { set_macro_value(1, v); }
void velvet_reverb_apply_tone_macro   (float v) { set_macro_value(2, v); }

/* ==== Morph alphas (per block, derived from per-tau / blocks-per-sec) ====
 * Block rate = fs_reverb / REVERB_BLOCK = 24000 / 16 = 1500 Hz.
 *   tau ≈ 1/(α × 1500). 0.0645 → ~10 ms (tap count), 0.0219 → ~30 ms
 *   (window), 0.00664 → ~100 ms (decay shape). */
#define ALPHA_COUNT_MORPH    0.0645f
#define ALPHA_WINDOW_MORPH   0.0219f
#define ALPHA_DECAY_MORPH    0.00664f
/* ==== Morph state ==== */
static float t0_count_morph;
static float t1_count_morph;
static float t2_count_morph;
static float t0_window_morph;       /* samples at fs_reverb */
static float t1_window_morph;
static float t2_window_morph;
static float t1_decay_morph;
static float t2_decay_morph;
static float t0_tap_comp_morph;
static float t1_tap_comp_morph;
static float t2_tap_comp_morph;
static float t0_full_energy;        /* sum(baseGain²) — set at IR-gen */
static float t1_full_energy;
static float t2_full_energy;

/* ==== Recirculation state ==== */
static float recirc_mod_phases[RECIRC_MOD_TOTAL] CCM_ATTR;
static float recirc_mod_rates [RECIRC_MOD_TOTAL] CCM_ATTR;
/* Per-stage recirc mod depth — scaled as a fraction of each stage's current
 * window so the same setting produces proportional pitch wobble on T0/T1/T2
 * regardless of their relative delay lengths. */
static float t0_rc_mod_depth_samples;
static float t1_rc_mod_depth_samples;
static float t2_rc_mod_depth_samples;
static float recirc_damp_coeff;
static float recirc_damp_t0 CCM_ATTR;
static float recirc_damp_t1 CCM_ATTR;
static float recirc_damp_t2 CCM_ATTR;
static float recirc_damp_global CCM_ATTR;   /* damping state for the T2->T0 global loop */

/* One-pole HPF in each feedback loop: hp_state is the low-passed feedback; the
 * high-passed output = fb - hp_state. Removes sub-RECIRC_HPF_FREQ_HZ buildup. */
static float recirc_hp_coeff;
static float recirc_hp_t0 CCM_ATTR;
static float recirc_hp_t1 CCM_ATTR;
static float recirc_hp_t2 CCM_ATTR;
static float recirc_hp_global CCM_ATTR;

/* Recirc-tap position ladders + per-block effective state. Each recirc tap
 * walks a discrete geometric ladder identical in shape to the main taps' —
 * compute_tap_states_mono picks the current rung per block based on the
 * morphed window, and produces a bell-shaped effGain that hits 0 at every
 * rung boundary. Replaces an earlier continuous IIR-smoothed offset that
 * audibly pitch-glided through the feedback loop on decay-macro sweeps. */
static uint16_t t0RecircLadder[T0_RECIRC_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint16_t t1RecircLadder[T1_RECIRC_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint16_t t2RecircLadder[T2_RECIRC_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t0RecircLadderCount[T0_RECIRC_TAPS] CCM_ATTR;
static uint8_t  t1RecircLadderCount[T1_RECIRC_TAPS] CCM_ATTR;
static uint8_t  t2RecircLadderCount[T2_RECIRC_TAPS] CCM_ATTR;
static uint16_t t0_recirc_effOff[T0_RECIRC_TAPS] CCM_ATTR;
static uint16_t t1_recirc_effOff[T1_RECIRC_TAPS] CCM_ATTR;
static uint16_t t2_recirc_effOff[T2_RECIRC_TAPS] CCM_ATTR;
static float    t0_recirc_effGain[T0_RECIRC_TAPS] CCM_ATTR;
static float    t1_recirc_effGain[T1_RECIRC_TAPS] CCM_ATTR;
static float    t2_recirc_effGain[T2_RECIRC_TAPS] CCM_ATTR;

/* ==== Output biquads ==== */
typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} Biquad;

static Biquad hpf_L CCM_ATTR;
static Biquad hpf_R CCM_ATTR;
static Biquad lpf_L CCM_ATTR;
static Biquad lpf_R CCM_ATTR;
static float last_lpf_hz = -1.0f;
static float last_hpf_hz = -1.0f;

#ifndef VELVET_REVERB_HOST
static inline void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
#endif

/* ==== Input / output ==== */
static int16_t input_buf_A[REVERB_BLOCK];
static int16_t input_buf_B[REVERB_BLOCK];
static int16_t *input_fill  = input_buf_A;
static int16_t *input_ready = input_buf_B;
static uint8_t  input_fill_idx = 0;
static int32_t push_acc = 0;
static uint8_t push_phase = 0;
static uint32_t block_write_idx = 0;
static volatile uint8_t block_ready = 0;

static int32_t accT0[REVERB_BLOCK] CCM_ATTR;
static int32_t accT1[REVERB_BLOCK] CCM_ATTR;
static int32_t accL [REVERB_BLOCK] CCM_ATTR;
static int32_t accR [REVERB_BLOCK] CCM_ATTR;

static int16_t outL[2][REVERB_OUT_BLOCK] CCM_ATTR;
static int16_t outR[2][REVERB_OUT_BLOCK] CCM_ATTR;
static volatile uint8_t output_buffer_id = 0;
/* Per-channel reader_buf. Each codec ISR samples output_buffer_id when its
 * idx wraps to 0, independently of the other. If finalize runs between the
 * two ISRs, L and R may briefly point at different buffers (one block of
 * skew, ~0.67 ms), but neither channel ever uses a stale reader_buf — and
 * neither depends on the two ISRs firing in lockstep. The earlier shared
 * wrap_pair approach broke when the two codec DMAs drifted: the channel
 * that wrapped twice between the other's wraps would skip its commit and
 * end up reading from a buffer two blocks old, producing audible constant-
 * sample artefacts on whichever channel had the faster clock. */
static uint8_t  reader_buf_L = 0;
static uint8_t  reader_buf_R = 0;
static uint8_t  out_idx_L = 0;
static uint8_t  out_idx_R = 0;

static float prev_out_L = 0.0f;   /* width-processed reverb-rate output (for upsample interp) */
static float prev_out_R = 0.0f;

static int16_t dma_scratch[2][REVERB_BLOCK] __attribute__((aligned(4)));
static uint8_t dma_buf_idx = 0;

/* ==== PRNG ==== */
static uint32_t prng_state = 0xDEADBEEF;
static uint32_t xorshift32(void)
{
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return x;
}

#ifdef VELVET_REVERB_HOST
/* Test hook (host only): reseed the tap-generation PRNG so two regenerations
 * produce an identical random layout. Lets the spectral harness vary one
 * parameter at a time with the tap positions held fixed. */
void host_reset_prng(void) { prng_state = 0xDEADBEEF; }
#endif

/* ==== Triangle wave LFO ==== Cheap stand-in for sin() on the recirc-mod
 * path. The mod runs sub-Hz so spectral content above the fundamental sits
 * far below the audible band, and the read-position-interp + damping LPF
 * smooth the corners anyway. Saves ~15 cycles per call vs. fastSin's
 * Taylor polynomial. Input phase in [-π, π]. */
static inline float triWave(float ph)
{
    float x = ph * (1.0f / PI_F);                /* x ∈ [-1, 1] */
    if (x >  0.5f) return  2.0f - 2.0f * x;
    if (x < -0.5f) return -2.0f - 2.0f * x;
    return 2.0f * x;
}

/* ==== Smoothstep helpers for fade curves ==== */
static inline float smoothstep01(float x)
{
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}
static inline float tap_gain_from_morph(float rank, float countMorph)
{
    return smoothstep01(countMorph - rank);
}
static inline float window_gain_from_offset(uint32_t offset, float windowSamples, float fadeSamples)
{
    return smoothstep01((windowSamples - (float)offset) / fadeSamples);
}
/* Per-tap decay envelope (frac ∈ [0,1] = effOff / window). exp falls to
 * ~1e-4 (-80 dB) across the window — steeper than a natural-room curve but
 * sounds right with the ladder keeping density constant; without the faster
 * falloff, late taps stay audible enough to muddy the tail.
 * Uses the precomputed exp_env_lut (filled in velvet_reverb_init). */
#define ENV_FRAC_BINS  32
static float exp_env_lut[ENV_FRAC_BINS] CCM_ATTR;

/* Largest plateau exponent (at shape = 2): 1 - frac^P. Higher = flatter hold
 * and a faster fade right at the end. */
#define DECAY_PLATEAU_P  10.0f

/* Plateau LUT for the shape ∈ (1,2] region. 2D: shape rows × frac columns,
 * plateau_lut[s][f] = 1 - (f/(N-1))^P(s), P(s) sweeping 1 (linear) ->
 * DECAY_PLATEAU_P across the rows. Filled with powf once at init so the per-tap
 * runtime cost is a bilinear lerp (no powf in recompute_eff_gains, which runs in
 * the main-loop block path — a per-tap powf there overran the block budget). */
#define PLATEAU_SHAPE_BINS  16
static float plateau_lut[PLATEAU_SHAPE_BINS * ENV_FRAC_BINS] CCM_ATTR;

/* shape: 0 = exponential, 1 = linear, 2 = "logarithmic" plateau (stays loud
 * across the window, quick fade only at the very end). 0..1 blends exp->linear
 * via exp_env_lut; 1..2 bilinearly interpolates plateau_lut. Continuous at
 * shape = 1 (linEnv == plateau row 0, P = 1). */
static inline float decay_envelope_from_frac(float frac, float shape)
{
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    if (shape <= 1.0f) {
        float bin_f = frac * (float)(ENV_FRAC_BINS - 1);
        int i = (int)bin_f;
        if (i > ENV_FRAC_BINS - 2) i = ENV_FRAC_BINS - 2;
        float t = bin_f - (float)i;
        float expEnv = exp_env_lut[i] + t * (exp_env_lut[i + 1] - exp_env_lut[i]);
        float linEnv = 1.0f - frac;
        return expEnv + shape * (linEnv - expEnv);
    }
    /* Bilinear interp into plateau_lut (shape-1 over the rows, frac over cols). */
    float sN = shape - 1.0f; if (sN > 1.0f) sN = 1.0f;
    float s_f = sN * (float)(PLATEAU_SHAPE_BINS - 1);
    int si = (int)s_f; if (si > PLATEAU_SHAPE_BINS - 2) si = PLATEAU_SHAPE_BINS - 2;
    float st = s_f - (float)si;
    float f_f = frac * (float)(ENV_FRAC_BINS - 1);
    int fi = (int)f_f; if (fi > ENV_FRAC_BINS - 2) fi = ENV_FRAC_BINS - 2;
    float ft = f_f - (float)fi;
    const float *r0 = &plateau_lut[si * ENV_FRAC_BINS];
    const float *r1 = &plateau_lut[(si + 1) * ENV_FRAC_BINS];
    float a = r0[fi] + ft * (r0[fi + 1] - r0[fi]);
    float b = r1[fi] + ft * (r1[fi + 1] - r1[fi]);
    return a + st * (b - a);
}

/* ==== Biquad cookbook coefficients ==== */
static void biquad_lpf(Biquad *bq, float freq, float fs)
{
    float w0 = TWO_PI_F * freq / fs;
    float alpha = sinf(w0) / (2.0f * 0.707f);
    float cosw = cosf(w0);
    float a0 = 1.0f + alpha;
    bq->b0 = ((1.0f - cosw) * 0.5f) / a0;
    bq->b1 = (1.0f - cosw) / a0;
    bq->b2 = ((1.0f - cosw) * 0.5f) / a0;
    bq->a1 = (-2.0f * cosw) / a0;
    bq->a2 = (1.0f - alpha) / a0;
}
static void biquad_hpf(Biquad *bq, float freq, float fs)
{
    float w0 = TWO_PI_F * freq / fs;
    float alpha = sinf(w0) / (2.0f * 0.707f);
    float cosw = cosf(w0);
    float a0 = 1.0f + alpha;
    bq->b0 = ((1.0f + cosw) * 0.5f) / a0;
    bq->b1 = -(1.0f + cosw) / a0;
    bq->b2 = ((1.0f + cosw) * 0.5f) / a0;
    bq->a1 = (-2.0f * cosw) / a0;
    bq->a2 = (1.0f - alpha) / a0;
}
__attribute__((always_inline)) static inline float biquad_process(Biquad *bq, float x)
{
    float y = bq->b0 * x + bq->z1;
    bq->z1 = bq->b1 * x - bq->a1 * y + bq->z2;
    bq->z2 = bq->b2 * x - bq->a2 * y;
    return y;
}

/* ==== DMA2 Stream1 helpers (T2 SDRAM prefetch) ==== */
#ifdef VELVET_REVERB_HOST
static inline void dma2_kick(const int16_t *src, int16_t *dst, uint32_t count) {
    memcpy(dst, src, count * sizeof(int16_t));
}
static inline void dma2_wait(void) { }
static inline void dma2_fetch_tap(uint32_t tapOffset, int16_t *dst)
{
    uint32_t idx = (block_write_idx - tapOffset) & T2_RING_MASK;
    if (idx + REVERB_BLOCK <= T2_RING_SAMPLES) {
        memcpy(dst, &t2_ring[idx], REVERB_BLOCK * sizeof(int16_t));
    } else {
        for (int i = 0; i < REVERB_BLOCK; i++)
            dst[i] = t2_ring[(idx + (uint32_t)i) & T2_RING_MASK];
    }
}
#else
/* DIAGNOSTIC: replace DMA2 prefetch with CPU memcpy. If channel-1 clicks
 * vanish with this, T2's DMA bus contention on SDRAM with channel 1's
 * audio reads is the source (they share the upper SDRAM half). Costs CPU
 * cycles (~16 word reads × 32 taps × 2 passes per block ≈ 1k cycles), so
 * expect slightly higher reverb_T2 timing but well under the deadline. */
#define DMA2_DISABLED_FOR_DIAGNOSTIC  0   /* revert: not the cause */

static inline void dma2_kick(const int16_t *src, int16_t *dst, uint32_t count)
{
#if DMA2_DISABLED_FOR_DIAGNOSTIC
    for (uint32_t i = 0; i < count; i++) dst[i] = src[i];
#else
    DMA2_Stream1->CR &= ~DMA_SxCR_EN;
    while (DMA2_Stream1->CR & DMA_SxCR_EN) {}
    DMA2->LIFCR = 0x00000F40;
    DMA2_Stream1->PAR  = (uint32_t)src;
    DMA2_Stream1->M0AR = (uint32_t)dst;
    DMA2_Stream1->NDTR = count;
    DMA2_Stream1->CR  |= DMA_SxCR_EN;
#endif
}
static inline void dma2_wait(void)
{
#if DMA2_DISABLED_FOR_DIAGNOSTIC
    /* memcpy is synchronous — nothing to wait for. */
#else
    while ((DMA2_Stream1->CR & DMA_SxCR_EN) && !(DMA2->LISR & DMA_LISR_TCIF1)) {}
    __DMB();
#endif
}
static inline void dma2_fetch_tap(uint32_t tapOffset, int16_t *dst)
{
    uint32_t idx = (block_write_idx - tapOffset) & T2_RING_MASK;
    if (idx + REVERB_BLOCK <= T2_RING_SAMPLES) {
        dma2_kick(&t2_ring[idx], dst, REVERB_BLOCK);
    } else {
        for (int i = 0; i < REVERB_BLOCK; i++)
            dst[i] = t2_ring[(idx + (uint32_t)i) & T2_RING_MASK];
    }
}
#endif

/* ==== Tap generation ==== */
#define TAP_MIN_OFFSET   128
#define TAP_MIN_GRID     16

/* Build one tap's ladder of candidate read positions. Returns the number of
 * rungs populated (≥ 1). Used by velvet_reverb_regenerate_taps after the
 * main offsets have been generated.
 *
 * Each rung is `parent × 0.25` plus a per-rung uniform jitter of ±25 % of the
 * band gap (preserves the velvet character at each rung without letting two
 * taps land on identical positions). For T2 the same jitter value is passed
 * in by the caller for both L and R chains so their divergence stays bounded
 * across rungs — see below. */
static int build_tap_ladder(uint16_t *ladder, uint32_t mainOffset, uint32_t minOffset)
{
    uint32_t cur = mainOffset;
    ladder[0] = (uint16_t)cur;
    int n = 1;
    while (n < MAX_LADDER_LEVELS) {
        uint32_t nxt = (cur * (uint32_t)LADDER_RATIO_Q15) >> 15;
        uint32_t bandGap = cur - nxt;
        uint32_t jitMag = bandGap >> 2;                /* ±25 % of band */
        int32_t jit = (jitMag > 0)
            ? (int32_t)(xorshift32() % (jitMag * 2u + 1u)) - (int32_t)jitMag : 0;
        int32_t signedNxt = (int32_t)nxt + jit;
        /* Round to even so the SIMD uint32 reads in do_t*_phase stay 4-byte
         * aligned. Same constraint generate_mono_stage applies to lad[0]. */
        signedNxt &= ~1;
        if (signedNxt < (int32_t)minOffset) break;
        if (signedNxt > (int32_t)cur - TAP_MIN_GRID)
            signedNxt = ((int32_t)cur - TAP_MIN_GRID) & ~1;
        if (signedNxt < (int32_t)minOffset) break;
        ladder[n] = (uint16_t)signedNxt;
        cur = (uint32_t)signedNxt;
        n++;
    }
    return n;
}

/* T2 variant — independent L/R chains, but the jitter draw is shared between
 * them at every rung. That bounds the L/R offset divergence (it inherits from
 * the per-channel L_0 jitter and tapers geometrically with the rung). With
 * an independent draw per channel, deep-rung divergence accumulates and is
 * heard as a ping-pong delay. */
/* Superseded by build_gapfill_ladders for the T2 main taps; kept for reference. */
static int build_tap_ladder_lr(uint16_t *ladderL, uint16_t *ladderR,
                               uint32_t mainOffsetL, uint32_t mainOffsetR) __attribute__((unused));
static int build_tap_ladder_lr(uint16_t *ladderL, uint16_t *ladderR,
                               uint32_t mainOffsetL, uint32_t mainOffsetR)
{
    uint32_t curL = mainOffsetL, curR = mainOffsetR;
    ladderL[0] = (uint16_t)curL;
    ladderR[0] = (uint16_t)curR;
    int n = 1;
    while (n < MAX_LADDER_LEVELS) {
        uint32_t nxtL = (curL * (uint32_t)LADDER_RATIO_Q15) >> 15;
        uint32_t nxtR = (curR * (uint32_t)LADDER_RATIO_Q15) >> 15;
        uint32_t bandGapL = curL - nxtL;
        uint32_t bandGapR = curR - nxtR;
        uint32_t bandGap  = (bandGapL < bandGapR) ? bandGapL : bandGapR;
        uint32_t jitMag   = bandGap >> 2;
        int32_t jit = (jitMag > 0)
            ? (int32_t)(xorshift32() % (jitMag * 2u + 1u)) - (int32_t)jitMag : 0;
        int32_t signedL = (int32_t)nxtL + jit;
        int32_t signedR = (int32_t)nxtR + jit;
        signedL &= ~1; signedR &= ~1;   /* even-rounded — DMA + SIMD alignment */
        if (signedL < TAP_MIN_OFFSET || signedR < TAP_MIN_OFFSET) break;
        if (signedL > (int32_t)curL - TAP_MIN_GRID) signedL = ((int32_t)curL - TAP_MIN_GRID) & ~1;
        if (signedR > (int32_t)curR - TAP_MIN_GRID) signedR = ((int32_t)curR - TAP_MIN_GRID) & ~1;
        if (signedL < TAP_MIN_OFFSET || signedR < TAP_MIN_OFFSET) break;
        ladderL[n] = (uint16_t)signedL;
        ladderR[n] = (uint16_t)signedR;
        curL = (uint32_t)signedL;
        curR = (uint32_t)signedR;
        n++;
    }
    return n;
}

/* Neighbour-spacing rank: lower rank = kept longer as density drops.
 * Iteratively removes the tap whose nearest neighbour is closest (= most
 * redundant). Early taps get a small "protected" bias so the first ~2 taps
 * never get rank-0'd out as long as ≥ 4 taps remain. O(N²) but only runs
 * at IR-gen time, so cheap in absolute terms. Mirrors `computeTapRanks`
 * in the JS prototype. */
#define MAX_STAGE_TAPS  64
static void compute_tap_ranks(const uint32_t *offsets, float *ranks, int count)
{
    if (count <= 0) return;
    if (count == 1) { ranks[0] = 0.0f; return; }

    int remaining[MAX_STAGE_TAPS];
    int removalOrder[MAX_STAGE_TAPS];
    int remainingLen = count;
    int removalLen = 0;
    for (int i = 0; i < count; i++) remaining[i] = i;

    int protectedCount = count < 2 ? count : 2;
    int protectedSurvivorFloor = count < 4 ? count : 4;
    uint32_t minOffset = offsets[0];
    uint32_t maxOffset = offsets[count - 1];
    float span = (float)((maxOffset > minOffset) ? (maxOffset - minOffset) : 1u);
    float meanGap = span / (float)(count > 1 ? count - 1 : 1);
    float earlyBiasScale = 0.15f * meanGap;

    while (remainingLen > 1) {
        int bestListIdx = -1;
        float bestScore = 1e30f;
        for (int listIdx = 0; listIdx < remainingLen; listIdx++) {
            int tapIdx = remaining[listIdx];
            if (tapIdx < protectedCount && remainingLen > protectedSurvivorFloor) {
                continue;
            }
            uint32_t tapPos = offsets[tapIdx];
            float leftDist = (listIdx > 0)
                ? (float)(tapPos - offsets[remaining[listIdx - 1]]) : 1e30f;
            float rightDist = (listIdx + 1 < remainingLen)
                ? (float)(offsets[remaining[listIdx + 1]] - tapPos) : 1e30f;
            float nearest = leftDist < rightDist ? leftDist : rightDist;
            float earlyBias = earlyBiasScale * (1.0f - ((float)(tapPos - minOffset)) / span);
            float removalScore = nearest + earlyBias;
            if (removalScore < bestScore) {
                bestScore = removalScore;
                bestListIdx = listIdx;
            }
        }
        if (bestListIdx < 0) bestListIdx = 0;
        removalOrder[removalLen++] = remaining[bestListIdx];
        for (int i = bestListIdx + 1; i < remainingLen; i++)
            remaining[i - 1] = remaining[i];
        remainingLen--;
    }
    if (remainingLen) removalOrder[removalLen++] = remaining[0];

    for (int orderIdx = 0; orderIdx < removalLen; orderIdx++) {
        int tapIdx = removalOrder[removalLen - 1 - orderIdx];
        ranks[tapIdx] = (float)orderIdx;
    }
}

#ifdef VELVET_REVERB_HOST
/* Test hook (host only): when 0, tap 0 is jittered like the others instead of
 * being pinned to TAP_MIN_OFFSET. Lets the spectral harness A/B whether the
 * pinned, triple-correlated early tap is responsible for a fixed spectral peak.
 * Firmware build always pins tap 0 (this symbol does not exist there). */
int host_pin_tap0 = 1;
#define TAP0_PINNED(j) ((j) == 0 && host_pin_tap0)
#else
#define TAP0_PINNED(j) ((j) == 0)
#endif

/* Generate one mono sparse-velvet stage. `apply_taper` controls whether the
 * cosine tail-taper is baked into the gains (T0 yes, T1/T2 no — they get the
 * decay envelope applied dynamically per-block instead). */
static void generate_mono_stage(uint32_t *outOffsets, int16_t *outGains, float *outRanks,
                                int *outCount, int wantTaps, int windowSamples,
                                int apply_taper)
{
    int n = wantTaps;
    int usable = windowSamples - TAP_MIN_OFFSET;
    if (usable < TAP_MIN_GRID) usable = TAP_MIN_GRID;
    int maxTaps = usable / TAP_MIN_GRID;
    if (n > maxTaps) n = maxTaps;

    int gridSize = usable / n;
    int jitterRange = gridSize - TAP_MIN_GRID;
    if (jitterRange < 1) jitterRange = 1;

    float norm = 1.0f / sqrtf((float)n);
    int taperLen = n * 15 / 100;
    if (taperLen < 1) taperLen = 1;

    for (int j = 0; j < n; j++) {
        /* Pin tap 0 to exactly TAP_MIN_OFFSET so every stage has a tap at
         * the very start of its window — guarantees an early reflection
         * regardless of window size. */
        int jit = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange);
        int pos = TAP_MIN_OFFSET + j * gridSize + jit;
        if (pos >= windowSamples) pos = TAP_MIN_OFFSET + j * gridSize;

        float sign = (xorshift32() & 1) ? 1.0f : -1.0f;
        float win = 1.0f;
        if (apply_taper && j >= n - taperLen) {
            int fromEnd = n - 1 - j;
            win = 0.5f * (1.0f - cosf(PI_F * (float)fromEnd / (float)taperLen));
        }

        outOffsets[j] = (uint32_t)(pos & ~1);
        outGains[j] = (int16_t)(sign * norm * win * TAP_GAIN_HEADROOM * 32767.0f);
    }
    *outCount = n;
    compute_tap_ranks(outOffsets, outRanks, n);
}

/* ---- Gap-fill ladder builder (MAIN taps) ----
 * Mirrors the JS app's buildGapFill. As the window shrinks, the tap at the
 * largest offset "falls off the end" and is reinserted into the largest interior
 * gap among the remaining taps — keeping them roughly equally spaced (velvet) and
 * the density up, instead of the geometric ×0.25 descent that scrambled the
 * spacing into [W/4, W). Produces a strictly-decreasing position sequence per tap
 * (rungs) — the exact structure compute_tap_states already consumes (silent
 * bell-crossfade at each boundary, one tap moving at a time).
 *
 * Mono (offR == NULL) for T0/T1; stereo for T2, where the L channel drives the
 * gap choice and R relocates into the same gap (between the same neighbour taps)
 * with a shared jitter draw, so L/R divergence stays bounded as before. Recirc
 * ladders keep the geometric build_tap_ladder (a feedback-delay chain, not a
 * window-filling layout). */
static inline int gapfill_clamp(int v, int lo, int hi, int minOffset)
{
    int a = lo + TAP_MIN_GRID, b = hi - TAP_MIN_GRID;
    if (b < a) { a = b = (lo + hi) / 2; }
    if (v < a) v = a;
    if (v > b) v = b;
    if (v < minOffset) v = minOffset;
    return v & ~1;   /* even — SIMD pair-read alignment in do_t*_phase */
}

static void build_gapfill_ladders(const uint32_t *offL, const uint32_t *offR, int count,
                                  uint16_t *ladL, uint16_t *ladR, uint8_t *ladderCount,
                                  int minOffset)
{
    int curL[MAX_STAGE_TAPS], curR[MAX_STAGE_TAPS], ord[MAX_STAGE_TAPS];
    int stereo = (offR != NULL);
    for (int k = 0; k < count; k++) {
        curL[k] = (int)offL[k];
        ladL[k * MAX_LADDER_LEVELS] = (uint16_t)offL[k];
        if (stereo) { curR[k] = (int)offR[k]; ladR[k * MAX_LADDER_LEVELS] = (uint16_t)offR[k]; }
        ladderCount[k] = 1;
    }
    int steps = count * MAX_LADDER_LEVELS;
    while (steps-- > 0) {
        /* tap at the largest current (L) position that still has ladder room */
        int endTap = -1, endPos = -1;
        for (int k = 0; k < count; k++) {
            if (ladderCount[k] >= MAX_LADDER_LEVELS) continue;
            if (curL[k] > endPos) { endPos = curL[k]; endTap = k; }
        }
        if (endTap < 0) break;
        /* sorted indices of the other taps below endPos */
        int m = 0;
        for (int k = 0; k < count; k++)
            if (k != endTap && curL[k] < endPos) ord[m++] = k;
        for (int a = 1; a < m; a++) {
            int vi = ord[a], vp = curL[vi], b = a - 1;
            while (b >= 0 && curL[ord[b]] > vp) { ord[b + 1] = ord[b]; b--; }
            ord[b + 1] = vi;
        }
        /* largest bottom/interior gap (top edge gap above the highest tap excluded
         * so the fallen tap densifies the interior rather than re-creating the end) */
        int bestSize = -1, bestLoIdx = -1, bestHiIdx = -1;
        int prevPos = minOffset, prevIdx = -1;
        for (int gi = 0; gi < m; gi++) {
            int hiIdx = ord[gi], hiPos = curL[hiIdx];
            int gap = hiPos - prevPos;
            if (gap > bestSize) { bestSize = gap; bestLoIdx = prevIdx; bestHiIdx = hiIdx; }
            prevPos = hiPos; prevIdx = hiIdx;
        }
        if (bestSize < 2 * TAP_MIN_GRID || bestHiIdx < 0) break;
        int loL = (bestLoIdx < 0) ? minOffset : curL[bestLoIdx];
        int hiL = curL[bestHiIdx];
        int span = hiL - loL; if (span > TAP_MIN_GRID * 8) span = TAP_MIN_GRID * 8;
        int jitMag = span >> 2;
        int jit = (jitMag > 0) ? (int)(xorshift32() % (uint32_t)(jitMag * 2 + 1)) - jitMag : 0;
        int newL = gapfill_clamp(((loL + hiL) / 2) + jit, loL, hiL, minOffset);
        if (newL >= curL[endTap]) break;   /* must descend */
        ladL[endTap * MAX_LADDER_LEVELS + ladderCount[endTap]] = (uint16_t)newL;
        if (stereo) {
            int loR = (bestLoIdx < 0) ? minOffset : curR[bestLoIdx];
            int hiR = curR[bestHiIdx];
            if (hiR < loR) { int t = hiR; hiR = loR; loR = t; }
            int newR = gapfill_clamp(((loR + hiR) / 2) + jit, loR, hiR, minOffset);
            ladR[endTap * MAX_LADDER_LEVELS + ladderCount[endTap]] = (uint16_t)newR;
            curR[endTap] = newR;
        }
        ladderCount[endTap]++;
        curL[endTap] = newL;
    }
}

void velvet_reverb_regenerate_taps(void)
{
    /* T0 — early. Taper baked in (cosine tail fade). */
    generate_mono_stage(t0TapOffsets, t0TapGains, t0TapRanks, &t0TapCount,
                        MAX_T0_TAPS, T0_WINDOW_MAX_SAMPLES, /*apply_taper=*/1);

    /* T1 — middle. No taper; decay envelope applied dynamically per-block. */
    generate_mono_stage(t1TapOffsets, t1TapGains, t1TapRanks, &t1TapCount,
                        MAX_T1_TAPS, T1_WINDOW_MAX_SAMPLES, /*apply_taper=*/0);

    /* T2 — stereo: independent L/R offsets, shared gain magnitude, no taper. */
    int t2Num = MAX_T2_TAPS;
    int usable2 = T2_DURATION_MAX_SAMPLES - TAP_MIN_OFFSET;
    if (usable2 < TAP_MIN_GRID) usable2 = TAP_MIN_GRID;
    int maxTaps2 = usable2 / TAP_MIN_GRID;
    if (t2Num > maxTaps2) t2Num = maxTaps2;

    int gridSize2 = usable2 / t2Num;
    int jitterRange2 = gridSize2 - TAP_MIN_GRID;
    if (jitterRange2 < 1) jitterRange2 = 1;

    float norm2 = 1.0f / sqrtf((float)t2Num);

    for (int j = 0; j < t2Num; j++) {
        /* Tap 0 pinned to TAP_MIN_OFFSET on both channels (see generate_mono_stage). */
        int jitL = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange2);
        int jitR = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange2);
        int posL = TAP_MIN_OFFSET + j * gridSize2 + jitL;
        int posR = TAP_MIN_OFFSET + j * gridSize2 + jitR;
        if (posL >= T2_DURATION_MAX_SAMPLES) posL = TAP_MIN_OFFSET + j * gridSize2;
        if (posR >= T2_DURATION_MAX_SAMPLES) posR = TAP_MIN_OFFSET + j * gridSize2;

        float sign = (xorshift32() & 1) ? 1.0f : -1.0f;

        t2TapOffsetsL[j] = (uint32_t)(posL & ~1);
        t2TapOffsetsR[j] = (uint32_t)(posR & ~1);
        t2TapGains[j] = (int16_t)(sign * norm2 * TAP_GAIN_HEADROOM * 32767.0f);
    }
    t2TapCount = t2Num;
    /* For ranking T2, use the L offsets (rank is consistent across channels
     * — both L and R share gain magnitude, only positions differ). */
    compute_tap_ranks(t2TapOffsetsL, t2TapRanks, t2Num);

    /* fullEnergy = sum of base-gain squared. Reference energy for the
     * loudness-preservation gain comp. Computed once per IR regen. */
    t0_full_energy = 0.0f;
    for (int i = 0; i < t0TapCount; i++) {
        float g = (float)t0TapGains[i];
        t0_full_energy += g * g;
    }
    t1_full_energy = 0.0f;
    for (int i = 0; i < t1TapCount; i++) {
        float g = (float)t1TapGains[i];
        t1_full_energy += g * g;
    }
    t2_full_energy = 0.0f;
    for (int i = 0; i < t2TapCount; i++) {
        float g = (float)t2TapGains[i];
        t2_full_energy += g * g;
    }

    /* ----------------------------------------------------------------
     * Build per-tap position ladders (MAIN taps) via gap-fill: as the window
     * shrinks the end tap is reinserted into the largest interior gap, keeping
     * the spacing velvet and the density up. Used at runtime by
     * compute_tap_states to migrate each tap as the window shrinks. */
    build_gapfill_ladders(t0TapOffsets, NULL, t0TapCount,
                          t0TapLadder, NULL, t0TapLadderCount, TAP_MIN_OFFSET);
    build_gapfill_ladders(t1TapOffsets, NULL, t1TapCount,
                          t1TapLadder, NULL, t1TapLadderCount, TAP_MIN_OFFSET);
    build_gapfill_ladders(t2TapOffsetsL, t2TapOffsetsR, t2TapCount,
                          t2TapLadderL, t2TapLadderR, t2TapLadderCount, TAP_MIN_OFFSET);

    /* Recirc tap ladders. Each recirc tap's L_0 sits at a fixed fraction of
     * its stage's max window; deeper rungs descend geometrically through
     * build_tap_ladder, getting per-rung jitter like the main taps.
     * compute_tap_states_mono will pick the active rung per block from the
     * morphed window. */
    {
        const float t0_recirc_l0_frac[T0_RECIRC_TAPS] = { 0.70f, 0.94f };
        const float t1_recirc_l0_frac[T1_RECIRC_TAPS] = { 0.70f, 0.94f };
        const float t2_recirc_l0_frac[T2_RECIRC_TAPS] = { 0.20f, 0.35f, 0.50f };
        for (int r = 0; r < T0_RECIRC_TAPS; r++) {
            uint32_t off = (uint32_t)((float)T0_WINDOW_MAX_SAMPLES * t0_recirc_l0_frac[r]);
            t0RecircLadderCount[r] = (uint8_t)build_tap_ladder(
                &t0RecircLadder[r * MAX_LADDER_LEVELS], off, RECIRC_MIN_OFFSET);
        }
        for (int r = 0; r < T1_RECIRC_TAPS; r++) {
            uint32_t off = (uint32_t)((float)T1_WINDOW_MAX_SAMPLES * t1_recirc_l0_frac[r]);
            t1RecircLadderCount[r] = (uint8_t)build_tap_ladder(
                &t1RecircLadder[r * MAX_LADDER_LEVELS], off, RECIRC_MIN_OFFSET);
        }
        for (int r = 0; r < T2_RECIRC_TAPS; r++) {
            uint32_t off = (uint32_t)((float)T2_DURATION_MAX_SAMPLES * t2_recirc_l0_frac[r]);
            t2RecircLadderCount[r] = (uint8_t)build_tap_ladder(
                &t2RecircLadder[r * MAX_LADDER_LEVELS], off, RECIRC_MIN_OFFSET);
        }
    }
}

/* Per-stage fade-zone minimum samples — keeps the bell smooth even at very
 * short window settings. Matches the JS prototype's per-stage floors. */
#define T0_FADE_MIN   16.0f
#define T1_FADE_MIN  128.0f
#define T2_FADE_MIN 1024.0f

/* Forward decl — defined further down, used here from init to seed recirc
 * effective state for block 0. */
static void compute_tap_states_mono(const uint16_t *ladder, const uint8_t *ladderCount,
                                    int count, float window, float fade,
                                    uint16_t *effOff, float *effWinGain);

/* ==== Init ==== */
void velvet_reverb_init(void)
{
    /* Apply default macro positions so the reverb_* targets reflect the
     * macro-derived values rather than the raw initialisers at the top of
     * this file. Density and Decay start at 0; Tone at 0.83 (from the JS
     * snapshot the bounds were captured from). */
    velvet_reverb_recompute_macros();

    {
        uint32_t *p = (uint32_t *)t0_ring;
        uint32_t *end = (uint32_t *)((uint8_t *)t0_ring + T0_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }
    {
        uint32_t *p = (uint32_t *)t1_ring;
        uint32_t *end = (uint32_t *)((uint8_t *)t1_ring + T1_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }
    {
        uint32_t *p = (uint32_t *)T2_RING_BASE;
        uint32_t *end = (uint32_t *)(T2_RING_BASE + T2_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }

    for (int i = 0; i < REVERB_BLOCK; i++) {
        accT0[i] = 0; accT1[i] = 0; accL[i] = 0; accR[i] = 0;
        input_buf_A[i] = 0; input_buf_B[i] = 0;
    }
    for (int i = 0; i < REVERB_OUT_BLOCK; i++) {
        outL[0][i] = 0; outR[0][i] = 0;
        outL[1][i] = 0; outR[1][i] = 0;
    }
    prev_out_L = 0; prev_out_R = 0;
    push_acc = 0; push_phase = 0;

    /* Biquads */
    hpf_L.z1 = hpf_L.z2 = 0.0f;  hpf_R.z1 = hpf_R.z2 = 0.0f;
    lpf_L.z1 = lpf_L.z2 = 0.0f;  lpf_R.z1 = lpf_R.z2 = 0.0f;
    biquad_hpf(&hpf_L, reverb_hpf_hz, 48000.0f);
    biquad_hpf(&hpf_R, reverb_hpf_hz, 48000.0f);
    biquad_lpf(&lpf_L, reverb_lpf_hz, 48000.0f);
    biquad_lpf(&lpf_R, reverb_lpf_hz, 48000.0f);
    last_hpf_hz = reverb_hpf_hz;
    last_lpf_hz = reverb_lpf_hz;

    /* Recirc state */
    recirc_damp_t0 = recirc_damp_t1 = recirc_damp_t2 = recirc_damp_global = 0.0f;
    recirc_hp_t0 = recirc_hp_t1 = recirc_hp_t2 = recirc_hp_global = 0.0f;
    /* Per-stage mod depths set fresh each block from window × percent — see
     * update_morph_state. Seed here so the first block has sane values. */
    {
        float frac = reverb_recirc_mod_depth_pct * 0.01f;
        t0_rc_mod_depth_samples = fmaxf(frac * t0_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
        t1_rc_mod_depth_samples = fmaxf(frac * t1_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
        t2_rc_mod_depth_samples = fmaxf(frac * t2_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
    }
    recirc_damp_coeff    = 1.0f - expf(-TWO_PI_F * RECIRC_DAMP_FREQ_HZ / (float)REVERB_FS_HZ);
    recirc_hp_coeff      = 1.0f - expf(-TWO_PI_F * RECIRC_HPF_FREQ_HZ / (float)REVERB_FS_HZ);
    {
        const float PHI = 1.6180339887498949f;
        float base_rate = TWO_PI_F * RECIRC_MOD_RATE_HZ / (float)REVERB_FS_HZ;
        for (int i = 0; i < RECIRC_MOD_TOTAL; i++) {
            float t = (float)i * PHI;
            t -= floorf(t);
            recirc_mod_rates[i] = base_rate * (0.6f + t * 1.8f);
            float ph = (float)i * (TWO_PI_F / (float)RECIRC_MOD_TOTAL);
            if (ph > PI_F) ph -= TWO_PI_F;
            recirc_mod_phases[i] = ph;
        }
    }

    /* Morph state — initialise to targets so the first block produces a
     * meaningful output (avoids "fading up from zero" silence on boot). */
    t0_count_morph  = reverb_t0_count_target;
    t1_count_morph  = reverb_t1_count_target;
    t2_count_morph  = reverb_t2_count_target;
    t0_window_morph = reverb_t0_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t1_window_morph = reverb_t1_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t2_window_morph = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t1_decay_morph  = reverb_t1_decay_shape;
    t2_decay_morph  = reverb_t2_decay_shape;
    t0_tap_comp_morph = 1.0f;
    t1_tap_comp_morph = 1.0f;
    t2_tap_comp_morph = 1.0f;

    /* Pre-compute the exponential envelope LUT — exp(-9.21034 × frac) gives
     * an 80 dB drop over the window. Linearly interpolated at lookup. */
    for (int i = 0; i < ENV_FRAC_BINS; i++) {
        float frac = (float)i / (float)(ENV_FRAC_BINS - 1);
        exp_env_lut[i] = expf(-9.21034f * frac);
    }

    /* Pre-compute the plateau LUT (shape 1..2 region) so the runtime envelope is
     * a bilinear lerp instead of a per-tap powf. powf only runs here, at init. */
    for (int s = 0; s < PLATEAU_SHAPE_BINS; s++) {
        float P = 1.0f + ((float)s / (float)(PLATEAU_SHAPE_BINS - 1)) * (DECAY_PLATEAU_P - 1.0f);
        for (int f = 0; f < ENV_FRAC_BINS; f++) {
            float frac = (float)f / (float)(ENV_FRAC_BINS - 1);
            plateau_lut[s * ENV_FRAC_BINS + f] = 1.0f - powf(frac, P);
        }
    }

    velvet_reverb_regenerate_taps();

    /* Seed recirc effective state for block 0 — update_morph_state would do
     * this on the first call, but pre-seeding avoids any first-block silence
     * if a recirc tap happens to start outside the active rung band. */
    {
        float t0_fade = t0_window_morph * 0.12f; if (t0_fade < T0_FADE_MIN) t0_fade = T0_FADE_MIN;
        float t1_fade = t1_window_morph * 0.12f; if (t1_fade < T1_FADE_MIN) t1_fade = T1_FADE_MIN;
        float t2_fade = t2_window_morph * 0.12f; if (t2_fade < T2_FADE_MIN) t2_fade = T2_FADE_MIN;
        compute_tap_states_mono(t0RecircLadder, t0RecircLadderCount, T0_RECIRC_TAPS,
                                t0_window_morph, t0_fade,
                                t0_recirc_effOff, t0_recirc_effGain);
        compute_tap_states_mono(t1RecircLadder, t1RecircLadderCount, T1_RECIRC_TAPS,
                                t1_window_morph, t1_fade,
                                t1_recirc_effOff, t1_recirc_effGain);
        compute_tap_states_mono(t2RecircLadder, t2RecircLadderCount, T2_RECIRC_TAPS,
                                t2_window_morph, t2_fade,
                                t2_recirc_effOff, t2_recirc_effGain);
    }

#ifndef VELVET_REVERB_HOST
    dwt_init();
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
    __DSB();
    DMA2_Stream1->CR = 0;
    while (DMA2_Stream1->CR & DMA_SxCR_EN) {}
    DMA2->LIFCR = 0x00000F40;
    DMA2_Stream1->CR = DMA_SxCR_DIR_1 | DMA_SxCR_MINC | DMA_SxCR_PINC
                     | (1 << 11) | (1 << 13) | (2 << 16);
    DMA2_Stream1->FCR = DMA_SxFCR_DMDIS | DMA_SxFCR_FTH;
#endif

    /* Seed effGains arrays so the first 3 blocks (before recompute round-
     * robin catches up) produce meaningful output rather than silence. */
    for (int i = 0; i < t0TapCount; i++) effGains_t0[i] = t0TapGains[i];
    for (int i = 0; i < t1TapCount; i++) effGains_t1[i] = t1TapGains[i];
    for (int i = 0; i < t2TapCount; i++) effGains_t2[i] = t2TapGains[i];

    /* Seed effOff arrays from each tap's main offset (rung 0). Without this,
     * the first block reads from offset 0 (near-zero delay = feedback path)
     * for every tap. */
    for (int i = 0; i < t0TapCount; i++)
        effOffT0[i] = (uint16_t)t0TapOffsets[i];
    for (int i = 0; i < t1TapCount; i++)
        effOffT1[i] = (uint16_t)t1TapOffsets[i];
    for (int i = 0; i < t2TapCount; i++) {
        effOffT2L[i] = (uint16_t)t2TapOffsetsL[i];
        effOffT2R[i] = (uint16_t)t2TapOffsetsR[i];
    }
}

/* ==== Saturate helpers ==== */

/* Soft knee in full-scale units (1.0 == full scale). Transparent (unity, like
 * the JS float path) for |x| <= SOFT_KNEE_T; above it a rational knee that is
 * C1-continuous at the threshold (value and slope match the linear region) and
 * asymptotes to ±1.0. Replaces the old hard clip so overs round off musically
 * instead of clipping harshly. */
static inline float soft_clip_unit(float x)
{
    float a = (x < 0.0f) ? -x : x;
    if (a <= SOFT_KNEE_T) return x;
    float k = 1.0f - SOFT_KNEE_T;
    float e = a - SOFT_KNEE_T;
    float y = SOFT_KNEE_T + k * e / (k + e);   /* -> 1.0, slope 1 at threshold */
    return (x < 0.0f) ? -y : y;
}

/* Q30 accumulator (Q15 sample × Q15 gain, summed) -> int16, soft-clipped at
 * full scale. 2^30 = full scale in the Q30 domain. Used at the stage bridges
 * and the output read. */
static inline int16_t soft_saturate_q15(int32_t x)
{
    float v = (float)x * (1.0f / 1073741824.0f);   /* 2^30 == full scale */
    return (int16_t)(soft_clip_unit(v) * 32767.0f);
}

/* Hard int16 clamp — kept for coefficient clamping (effGains) and anywhere a
 * true hard limit is wanted. */
static inline int16_t clamp_f_to_int16(float v)
{
    if (v > 32767.0f) return 32767;
    if (v < -32767.0f) return -32767;
    return (int16_t)v;
}

/* Soft int16 clamp for signal-path values already in int16 scale (recirc
 * feedback writes, final output). Same knee as soft_saturate_q15. */
static inline int16_t soft_clip_int16(float v)
{
    return (int16_t)(soft_clip_unit(v * (1.0f / 32768.0f)) * 32767.0f);
}

/* ==== Per-block effGains recompute (ladder-driven) ====
 *
 * Each tap migrates one rung down each time the window shrinks past its
 * current rung. compute_tap_states picks the rung whose band contains the
 * current window and computes a bell-shaped window-gain that is 0 at every
 * rung boundary, so the discrete position swap happens at gain = 0 (no click).
 *
 * Recompute pass:
 *   1. compute_tap_states → effOff[], effWinGain[]
 *   2. for each tap: pre-comp effGain = baseGain × densMorph × winGain × env
 *   3. sum (pre-comp²) → comp = sqrt(full_energy / energy), morphed
 *   4. final effGain = clamp_int16(pre-comp × comp_morph)
 */

/* Compute the per-tap "active" fade by rank: only the boundary tap is partial
 * as count_morph crosses an integer; the rest are 1.0 or 0.0. */
static inline float fade_from_rank(float rank, float count_morph)
{
    return smoothstep01(count_morph - rank);
}

/* Walk each tap's ladder to find which rung's band contains the current
 * window. Picks the smaller-offset rung (lower edge of the band) as the
 * effective read position, with a bell-shaped gain that is 0 at both the
 * upper and lower rung boundaries — so the discrete position swap at each
 * boundary happens in silence. */
static void compute_tap_states_mono(const uint16_t *ladder, const uint8_t *ladderCount,
                                    int count, float window, float fade,
                                    uint16_t *effOff, float *effWinGain)
{
    float invFade = (fade > 1e-6f) ? 1.0f / fade : 0.0f;
    for (int k = 0; k < count; k++) {
        const uint16_t *lad = &ladder[k * MAX_LADDER_LEVELS];
        int nLevels = ladderCount[k];
        float L0 = (float)lad[0];

        if (window >= L0) {
            /* Above the topmost rung — ramp in from 0 at window = L_0. */
            effOff[k] = lad[0];
            effWinGain[k] = smoothstep01((window - L0) * invFade);
            continue;
        }

        /* Walk rungs to find the band containing the window. */
        int foundI = -1;
        for (int i = 0; i < nLevels - 1; i++) {
            if (window > (float)lad[i + 1]) { foundI = i; break; }
        }
        if (foundI < 0) {
            /* Below the deepest rung — silent. */
            effOff[k] = lad[nLevels - 1];
            effWinGain[k] = 0.0f;
            continue;
        }
        float upper = (float)lad[foundI];
        float lower = (float)lad[foundI + 1];
        float sUp = smoothstep01((upper - window) * invFade);
        float sDn = smoothstep01((window - lower) * invFade);
        effOff[k] = lad[foundI + 1];
        effWinGain[k] = (sUp < sDn) ? sUp : sDn;
    }
}

/* T2 variant — band check uses the L ladder; R picks the matching rung
 * index. L and R ladders share their jitter at every rung, so they index
 * the same band, just with different absolute read positions for stereo
 * decorrelation. */
static void compute_tap_states_lr(const uint16_t *ladderL, const uint16_t *ladderR,
                                  const uint8_t *ladderCount, int count,
                                  float window, float fade,
                                  uint16_t *effOffL, uint16_t *effOffR,
                                  float *effWinGain)
{
    float invFade = (fade > 1e-6f) ? 1.0f / fade : 0.0f;
    for (int k = 0; k < count; k++) {
        const uint16_t *ladL = &ladderL[k * MAX_LADDER_LEVELS];
        const uint16_t *ladR = &ladderR[k * MAX_LADDER_LEVELS];
        int nLevels = ladderCount[k];
        float L0 = (float)ladL[0];

        if (window >= L0) {
            effOffL[k] = ladL[0];
            effOffR[k] = ladR[0];
            effWinGain[k] = smoothstep01((window - L0) * invFade);
            continue;
        }
        int foundI = -1;
        for (int i = 0; i < nLevels - 1; i++) {
            if (window > (float)ladL[i + 1]) { foundI = i; break; }
        }
        if (foundI < 0) {
            effOffL[k] = ladL[nLevels - 1];
            effOffR[k] = ladR[nLevels - 1];
            effWinGain[k] = 0.0f;
            continue;
        }
        float upper = (float)ladL[foundI];
        float lower = (float)ladL[foundI + 1];
        float sUp = smoothstep01((upper - window) * invFade);
        float sDn = smoothstep01((window - lower) * invFade);
        effOffL[k] = ladL[foundI + 1];
        effOffR[k] = ladR[foundI + 1];
        effWinGain[k] = (sUp < sDn) ? sUp : sDn;
    }
}

static void recompute_eff_gains_t0(void)
{
    int count = t0TapCount;
    if (count <= 0) return;

    float window = t0_window_morph;
    float fade = window * 0.12f;
    if (fade < T0_FADE_MIN) fade = T0_FADE_MIN;

    float effWin[MAX_T0_TAPS];
    compute_tap_states_mono(t0TapLadder, t0TapLadderCount, count, window, fade,
                            effOffT0, effWin);

    /* T0 has no decay envelope — taper is baked into baseGain. */
    float pre[MAX_T0_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float densMorph = fade_from_rank(t0TapRanks[t], t0_count_morph);
        float g = (float)t0TapGains[t] * densMorph * effWin[t];
        pre[t] = g;
        energy += g * g;
    }

    float comp = 1.0f;
    if (energy > 1e-3f && t0_full_energy > 1e-3f) {
        comp = sqrtf(t0_full_energy / energy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t0_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t0_tap_comp_morph);

    for (int t = 0; t < count; t++) {
        effGains_t0[t] = clamp_f_to_int16(pre[t] * t0_tap_comp_morph);
    }
}

static void recompute_eff_gains_t1(void)
{
    int count = t1TapCount;
    if (count <= 0) return;

    float window = t1_window_morph;
    float fade = window * 0.12f;
    if (fade < T1_FADE_MIN) fade = T1_FADE_MIN;
    float invWindow = (window > 1e-6f) ? 1.0f / window : 0.0f;

    float effWin[MAX_T1_TAPS];
    compute_tap_states_mono(t1TapLadder, t1TapLadderCount, count, window, fade,
                            effOffT1, effWin);

    float pre[MAX_T1_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float densMorph = fade_from_rank(t1TapRanks[t], t1_count_morph);
        float frac = (float)effOffT1[t] * invWindow;
        float env  = decay_envelope_from_frac(frac, t1_decay_morph);
        float g = (float)t1TapGains[t] * densMorph * effWin[t] * env;
        pre[t] = g;
        energy += g * g;
    }

    float comp = 1.0f;
    if (energy > 1e-3f && t1_full_energy > 1e-3f) {
        comp = sqrtf(t1_full_energy / energy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t1_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t1_tap_comp_morph);

    for (int t = 0; t < count; t++) {
        effGains_t1[t] = clamp_f_to_int16(pre[t] * t1_tap_comp_morph);
    }
}

static void recompute_eff_gains_t2(void)
{
    int count = t2TapCount;
    if (count <= 0) return;

    float window = t2_window_morph;
    float fade = window * 0.12f;
    if (fade < T2_FADE_MIN) fade = T2_FADE_MIN;
    float invWindow = (window > 1e-6f) ? 1.0f / window : 0.0f;

    float effWin[MAX_T2_TAPS];
    compute_tap_states_lr(t2TapLadderL, t2TapLadderR, t2TapLadderCount, count,
                          window, fade, effOffT2L, effOffT2R, effWin);

    float pre[MAX_T2_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float densMorph = fade_from_rank(t2TapRanks[t], t2_count_morph);
        float frac = (float)effOffT2L[t] * invWindow;
        float env  = decay_envelope_from_frac(frac, t2_decay_morph);
        float g = (float)t2TapGains[t] * densMorph * effWin[t] * env;
        pre[t] = g;
        energy += g * g;
    }

    float comp = 1.0f;
    if (energy > 1e-3f && t2_full_energy > 1e-3f) {
        comp = sqrtf(t2_full_energy / energy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t2_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t2_tap_comp_morph);

    for (int t = 0; t < count; t++) {
        effGains_t2[t] = clamp_f_to_int16(pre[t] * t2_tap_comp_morph);
    }
}

/* ==== Per-block morph state update (inside the block timer — cheap) ====
 *
 * Advances morph state toward targets (one-pole IIR) and refreshes recirc
 * tap offsets to track the morphed windows. Both are cheap (~5 µs CPU).
 * The expensive effGains recompute is done OUTSIDE the block timer — see
 * background_eff_gains_update() below. */
static void update_morph_state(void)
{
    /* Throttled macro recompute. apply_*_macro is called from params.c at
     * main-loop rate (~1 kHz); recomputing the param table there fires
     * ~14 powf calls (~17 µs) on every iteration just to track 12-bit ADC
     * jitter. The morph IIRs immediately downstream have a ~30 ms tau, so
     * anything faster than ~100 Hz updates the morph targets in a way the
     * smoother can't track. Run at most once per 16 audio blocks (~11 ms)
     * and only when a macro value actually moved past the dead-band. */
    static uint8_t macro_throttle_counter = 0;
    if (++macro_throttle_counter >= 16) {
        macro_throttle_counter = 0;
        if (macros_dirty) {
            macros_dirty = 0;
            velvet_reverb_recompute_macros();
        }
    }

    t0_count_morph  += ALPHA_COUNT_MORPH  * (reverb_t0_count_target - t0_count_morph);
    t1_count_morph  += ALPHA_COUNT_MORPH  * (reverb_t1_count_target - t1_count_morph);
    t2_count_morph  += ALPHA_COUNT_MORPH  * (reverb_t2_count_target - t2_count_morph);
    float t0_win_target = reverb_t0_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t1_win_target = reverb_t1_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t2_win_target = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t0_window_morph += ALPHA_WINDOW_MORPH * (t0_win_target - t0_window_morph);
    t1_window_morph += ALPHA_WINDOW_MORPH * (t1_win_target - t1_window_morph);
    t2_window_morph += ALPHA_WINDOW_MORPH * (t2_win_target - t2_window_morph);
    t1_decay_morph  += ALPHA_DECAY_MORPH  * (reverb_t1_decay_shape - t1_decay_morph);
    t2_decay_morph  += ALPHA_DECAY_MORPH  * (reverb_t2_decay_shape - t2_decay_morph);

    /* Recirc tap effective state — discrete ladder migration with a
     * bell-shaped gain that's 0 at every rung boundary. The discrete
     * position swaps happen in silence (effGain = 0 at the boundary), so
     * no continuous offset sweep, so no pitch glide in the feedback loop
     * during a decay-macro sweep. Mirrors how the main taps migrate. */
    float t0_fade = t0_window_morph * 0.12f; if (t0_fade < T0_FADE_MIN) t0_fade = T0_FADE_MIN;
    float t1_fade = t1_window_morph * 0.12f; if (t1_fade < T1_FADE_MIN) t1_fade = T1_FADE_MIN;
    float t2_fade = t2_window_morph * 0.12f; if (t2_fade < T2_FADE_MIN) t2_fade = T2_FADE_MIN;
    compute_tap_states_mono(t0RecircLadder, t0RecircLadderCount, T0_RECIRC_TAPS,
                            t0_window_morph, t0_fade,
                            t0_recirc_effOff, t0_recirc_effGain);
    compute_tap_states_mono(t1RecircLadder, t1RecircLadderCount, T1_RECIRC_TAPS,
                            t1_window_morph, t1_fade,
                            t1_recirc_effOff, t1_recirc_effGain);
    compute_tap_states_mono(t2RecircLadder, t2RecircLadderCount, T2_RECIRC_TAPS,
                            t2_window_morph, t2_fade,
                            t2_recirc_effOff, t2_recirc_effGain);

    /* Per-stage recirc mod depths in samples — proportional to each stage's
     * window so the same % setting produces a consistent feel across T0/T1/T2. */
    float mod_frac = reverb_recirc_mod_depth_pct * 0.01f;
    t0_rc_mod_depth_samples = fmaxf(mod_frac * t0_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
    t1_rc_mod_depth_samples = fmaxf(mod_frac * t1_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
    t2_rc_mod_depth_samples = fmaxf(mod_frac * t2_window_morph, RECIRC_MOD_DEPTH_MIN_SAMPLES);
}

/* ==== Background effGains recompute (outside the block timer) ====
 *
 * Two conflicting requirements:
 *   1. To avoid audible step artefacts in the convolution output the
 *      effGains must be re-derived from the current morph state often
 *      (every block is ideal).
 *   2. The full 3-stage recompute is ~90 µs of main-loop wall time —
 *      enough to push the main-loop iteration past the inter-block
 *      period when run every block, causing drops.
 *
 * Resolution: detect target changes with per-parameter dead-bands and
 * recompute every block for ~40 ms after each change (the morph IIR's
 * settle window). When the user is not turning knobs, recompute pauses
 * and effGains sit static — same shape as Phase 1's behaviour, no drops.
 * A slow periodic refresh (every ~67 ms) corrects any morph-state drift
 * during long steady periods. */

#define EFFGAINS_SETTLE_BLOCKS    60    /* ~40 ms of recompute after a target change */
#define EFFGAINS_PERIODIC_BLOCKS  100   /* ~67 ms safety refresh interval */

/* Per-target dead-bands. Set to "comfortably above ADC/morph noise floor"
 * so genuine user gestures register but pot jitter doesn't. */
#define DB_COUNT_TARGET    0.10f
#define DB_WINDOW_TARGET   0.50f      /* ms */
#define DB_DURATION_TARGET 0.01f      /* seconds */
#define DB_DECAY_TARGET    0.005f

static float last_t0_cnt_target = -1e9f, last_t1_cnt_target = -1e9f, last_t2_cnt_target = -1e9f;
static float last_t0_win_target = -1e9f, last_t1_win_target = -1e9f, last_t2_dur_target = -1e9f;
static float last_t1_dec_target = -1e9f, last_t2_dec_target = -1e9f;
static uint16_t effgains_settle_counter = 0;
static uint16_t effgains_periodic_counter = 0;

static void background_eff_gains_update(void)
{
    /* Watch user-facing targets for movement. Each target has a dead-band
     * tuned to ignore ADC/smoothing jitter while catching deliberate moves. */
    int changed = 0;
    if (fabsf(reverb_t0_count_target - last_t0_cnt_target) > DB_COUNT_TARGET) {
        last_t0_cnt_target = reverb_t0_count_target; changed = 1;
    }
    if (fabsf(reverb_t1_count_target - last_t1_cnt_target) > DB_COUNT_TARGET) {
        last_t1_cnt_target = reverb_t1_count_target; changed = 1;
    }
    if (fabsf(reverb_t2_count_target - last_t2_cnt_target) > DB_COUNT_TARGET) {
        last_t2_cnt_target = reverb_t2_count_target; changed = 1;
    }
    if (fabsf(reverb_t0_window_ms_target - last_t0_win_target) > DB_WINDOW_TARGET) {
        last_t0_win_target = reverb_t0_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t1_window_ms_target - last_t1_win_target) > DB_WINDOW_TARGET) {
        last_t1_win_target = reverb_t1_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t2_duration_s_target - last_t2_dur_target) > DB_DURATION_TARGET) {
        last_t2_dur_target = reverb_t2_duration_s_target; changed = 1;
    }
    if (fabsf(reverb_t1_decay_shape - last_t1_dec_target) > DB_DECAY_TARGET) {
        last_t1_dec_target = reverb_t1_decay_shape; changed = 1;
    }
    if (fabsf(reverb_t2_decay_shape - last_t2_dec_target) > DB_DECAY_TARGET) {
        last_t2_dec_target = reverb_t2_decay_shape; changed = 1;
    }

    if (changed) effgains_settle_counter = EFFGAINS_SETTLE_BLOCKS;

    /* Slow periodic refresh — catches morph-state drift during long
     * stationary periods so effGains don't fossilise out of sync. */
    if (++effgains_periodic_counter >= EFFGAINS_PERIODIC_BLOCKS) {
        effgains_periodic_counter = 0;
        if (effgains_settle_counter == 0) effgains_settle_counter = 1;
    }

    if (effgains_settle_counter == 0) return;
    effgains_settle_counter--;

    /* Rotate stages across successive blocks rather than running all three
     * per block. With 3 stages and a ~60-block settle window after a target
     * change, each stage still gets ~20 recomputes during the 40-ms settle
     * — every ~2 ms, well below the 30-ms morph IIR tau. Caps per-block BG
     * cost at one stage's recompute (~75 µs instead of ~225) so the main
     * loop's (block + bg_eff) total stays well under the 666 µs deadline
     * even while every macro is being CV-modulated. */
    static int stage_round_robin = 0;
    switch (stage_round_robin) {
        case 0: recompute_eff_gains_t0(); break;
        case 1: recompute_eff_gains_t1(); break;
        case 2: recompute_eff_gains_t2(); break;
        default: break;
    }
    stage_round_robin = (stage_round_robin + 1) % 3;
}

/* ==== Phase handlers ==== */

static void do_input_write_t0(void)
{
    /* Attenuate the input into the cascade by REVERB_HEADROOM so the whole
     * int16 chain runs cooler; the output stage makes it back up. */
    for (int i = 0; i < REVERB_BLOCK; i++) {
        uint32_t wi = ring_write_idx + (uint32_t)i;
        t0_ring[wi & T0_RING_MASK] = (int16_t)((float)input_ready[i] * REVERB_HEADROOM);
    }
    block_write_idx = ring_write_idx;
    ring_write_idx += REVERB_BLOCK;

    for (int i = 0; i < REVERB_BLOCK; i++) accT0[i] = 0;
}

static void do_t0_phase(void)
{
    int tapEnd = t0TapCount;
    for (int t = 0; t < tapEnd; t++) {
        uint32_t gain = (uint32_t)(int32_t)effGains_t0[t];
        uint32_t base = (block_write_idx - (uint32_t)effOffT0[t]) & T0_RING_MASK;

        uint32_t until_wrap = T0_RING_SAMPLES - base;
        uint32_t n1 = (until_wrap < (uint32_t)REVERB_BLOCK) ? until_wrap : (uint32_t)REVERB_BLOCK;

        const int16_t *src = &t0_ring[base];
        int i = 0;
        for (uint32_t k = 0; k < n1; k += 2, i += 2) {
            uint32_t s_pair = *((const u32_alias *)(src + k));
            int32_t p0 = smulbb(s_pair, gain);
            int32_t p1 = smultb(s_pair, gain);
            accT0[i]   = qadd_sat(accT0[i],   p0);
            accT0[i+1] = qadd_sat(accT0[i+1], p1);
        }
        if (n1 < (uint32_t)REVERB_BLOCK) {
            src = &t0_ring[0];
            uint32_t n2 = (uint32_t)REVERB_BLOCK - n1;
            for (uint32_t k = 0; k < n2; k += 2, i += 2) {
                uint32_t s_pair = *((const u32_alias *)(src + k));
                int32_t p0 = smulbb(s_pair, gain);
                int32_t p1 = smultb(s_pair, gain);
                accT0[i]   = qadd_sat(accT0[i],   p0);
                accT0[i+1] = qadd_sat(accT0[i+1], p1);
            }
        }
    }
}

static void do_t0_recirc(void)
{
    float amt = reverb_t0_recirc;
    if (amt > RECIRC_AMOUNT_CLAMP) amt = RECIRC_AMOUNT_CLAMP;
    if (amt <= 0.0f) return;

    /* Bell-weighted sum: at rung-boundary transitions effGain dips to 0,
     * cleanly removing that tap from the average until it re-engages on the
     * next rung. Dividing by sum(gain) keeps total loudness constant when
     * any subset is mid-transition. */
    float wSum = t0_recirc_effGain[0] + t0_recirc_effGain[1];
    if (wSum < 1e-6f) return;
    float invW = 1.0f / wSum;

    /* Per-block precompute: the recirc mod is sub-Hz so phase moves <0.005
     * rad over a 16-sample block. Compute modOff (and therefore iBase + frac)
     * once per tap per block, then the per-sample inner loop is just two
     * ring reads + a lerp. Drops fastSin/floorf out of the hot path. */
    int32_t iBase[T0_RECIRC_TAPS];
    float   frac [T0_RECIRC_TAPS];
    for (int r = 0; r < T0_RECIRC_TAPS; r++) {
        if (t0_recirc_effGain[r] <= 0.0f) continue;
        int pidx = r;
        recirc_mod_phases[pidx] += recirc_mod_rates[pidx] * (float)REVERB_BLOCK;
        if (recirc_mod_phases[pidx] >  PI_F) recirc_mod_phases[pidx] -= TWO_PI_F;
        if (recirc_mod_phases[pidx] < -PI_F) recirc_mod_phases[pidx] += TWO_PI_F;
        float modOff = (float)t0_recirc_effOff[r]
                     + t0_rc_mod_depth_samples * triWave(recirc_mod_phases[pidx]);
        float pos0  = (float)block_write_idx - modOff;
        int32_t iP  = (int32_t)floorf(pos0);
        iBase[r]    = iP;
        frac [r]    = pos0 - (float)iP;
    }

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float rcSum = 0.0f;
        for (int r = 0; r < T0_RECIRC_TAPS; r++) {
            float g = t0_recirc_effGain[r];
            if (g <= 0.0f) continue;
            uint32_t idx0 = ((uint32_t)(iBase[r] + n))     & T0_RING_MASK;
            uint32_t idx1 = ((uint32_t)(iBase[r] + n + 1)) & T0_RING_MASK;
            float a = (float)t0_ring[idx0];
            float b = (float)t0_ring[idx1];
            rcSum += g * (a + frac[r] * (b - a));
        }
        rcSum *= invW;
        recirc_damp_t0 += recirc_damp_coeff * (rcSum - recirc_damp_t0);
        float shelf = recirc_damp_t0 + reverb_recirc_damp_hf_gain * (rcSum - recirc_damp_t0);
        recirc_hp_t0 += recirc_hp_coeff * (shelf - recirc_hp_t0);
        float fb = shelf - recirc_hp_t0;   /* high-shelf damping + 100 Hz HPF */

        uint32_t wi = (block_write_idx + (uint32_t)n) & T0_RING_MASK;
        float current = (float)t0_ring[wi];
        t0_ring[wi] = soft_clip_int16(current + amt * fb);
    }
}

static void do_bridge_t0_to_t1(void)
{
    for (int i = 0; i < REVERB_BLOCK; i++) {
        uint32_t wi = block_write_idx + (uint32_t)i;
        t1_ring[wi & T1_RING_MASK] = soft_saturate_q15(accT0[i]);
        accT1[i] = 0;
    }
}

static void do_t1_phase(void)
{
    int tapEnd = t1TapCount;
    for (int t = 0; t < tapEnd; t++) {
        uint32_t gain = (uint32_t)(int32_t)effGains_t1[t];
        uint32_t base = (block_write_idx - (uint32_t)effOffT1[t]) & T1_RING_MASK;

        uint32_t until_wrap = T1_RING_SAMPLES - base;
        uint32_t n1 = (until_wrap < (uint32_t)REVERB_BLOCK) ? until_wrap : (uint32_t)REVERB_BLOCK;

        const int16_t *src = &t1_ring[base];
        int i = 0;
        for (uint32_t k = 0; k < n1; k += 2, i += 2) {
            uint32_t s_pair = *((const u32_alias *)(src + k));
            int32_t p0 = smulbb(s_pair, gain);
            int32_t p1 = smultb(s_pair, gain);
            accT1[i]   = qadd_sat(accT1[i],   p0);
            accT1[i+1] = qadd_sat(accT1[i+1], p1);
        }
        if (n1 < (uint32_t)REVERB_BLOCK) {
            src = &t1_ring[0];
            uint32_t n2 = (uint32_t)REVERB_BLOCK - n1;
            for (uint32_t k = 0; k < n2; k += 2, i += 2) {
                uint32_t s_pair = *((const u32_alias *)(src + k));
                int32_t p0 = smulbb(s_pair, gain);
                int32_t p1 = smultb(s_pair, gain);
                accT1[i]   = qadd_sat(accT1[i],   p0);
                accT1[i+1] = qadd_sat(accT1[i+1], p1);
            }
        }
    }
}

static void do_t1_recirc(void)
{
    float amt = reverb_t1_recirc;
    if (amt > RECIRC_AMOUNT_CLAMP) amt = RECIRC_AMOUNT_CLAMP;
    if (amt <= 0.0f) return;

    float wSum = t1_recirc_effGain[0] + t1_recirc_effGain[1];
    if (wSum < 1e-6f) return;
    float invW = 1.0f / wSum;

    int32_t iBase[T1_RECIRC_TAPS];
    float   frac [T1_RECIRC_TAPS];
    for (int r = 0; r < T1_RECIRC_TAPS; r++) {
        if (t1_recirc_effGain[r] <= 0.0f) continue;
        int pidx = T0_RECIRC_TAPS + r;
        recirc_mod_phases[pidx] += recirc_mod_rates[pidx] * (float)REVERB_BLOCK;
        if (recirc_mod_phases[pidx] >  PI_F) recirc_mod_phases[pidx] -= TWO_PI_F;
        if (recirc_mod_phases[pidx] < -PI_F) recirc_mod_phases[pidx] += TWO_PI_F;
        float modOff = (float)t1_recirc_effOff[r]
                     + t1_rc_mod_depth_samples * triWave(recirc_mod_phases[pidx]);
        float pos0  = (float)block_write_idx - modOff;
        int32_t iP  = (int32_t)floorf(pos0);
        iBase[r]    = iP;
        frac [r]    = pos0 - (float)iP;
    }

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float rcSum = 0.0f;
        for (int r = 0; r < T1_RECIRC_TAPS; r++) {
            float g = t1_recirc_effGain[r];
            if (g <= 0.0f) continue;
            uint32_t idx0 = ((uint32_t)(iBase[r] + n))     & T1_RING_MASK;
            uint32_t idx1 = ((uint32_t)(iBase[r] + n + 1)) & T1_RING_MASK;
            float a = (float)t1_ring[idx0];
            float b = (float)t1_ring[idx1];
            rcSum += g * (a + frac[r] * (b - a));
        }
        rcSum *= invW;
        recirc_damp_t1 += recirc_damp_coeff * (rcSum - recirc_damp_t1);
        float shelf = recirc_damp_t1 + reverb_recirc_damp_hf_gain * (rcSum - recirc_damp_t1);
        recirc_hp_t1 += recirc_hp_coeff * (shelf - recirc_hp_t1);
        float fb = shelf - recirc_hp_t1;   /* high-shelf damping + 100 Hz HPF */

        uint32_t wi = (block_write_idx + (uint32_t)n) & T1_RING_MASK;
        float current = (float)t1_ring[wi];
        t1_ring[wi] = soft_clip_int16(current + amt * fb);
    }
}

static void do_bridge_t1_to_t2(void)
{
    for (int i = 0; i < REVERB_BLOCK; i += 2) {
        int16_t s0 = soft_saturate_q15(accT1[i]);
        int16_t s1 = soft_saturate_q15(accT1[i + 1]);
        uint32_t packed = ((uint32_t)(uint16_t)s1 << 16) | (uint32_t)(uint16_t)s0;
        uint32_t wi = block_write_idx + (uint32_t)i;
        *((volatile u32_alias *)&t2_ring[wi & T2_RING_MASK]) = packed;
    }
    for (int i = 0; i < REVERB_BLOCK; i++) { accL[i] = 0; accR[i] = 0; }
}

/* Largest T2 effective tap offset that still carries audible energy (effGain
 * within ~5% of the loudest tap), updated each block by do_t2_phase. The global
 * recirc bounds its read delays to this so it overlaps the real tail instead of
 * reading past the last active tap (a distinct echo when T2 density/decay shrinks
 * the tail). 1-block latency (global recirc runs before do_t2_phase). */
static float t2_active_extent = 0.0f;

static void do_t2_phase(void)
{
    int count = t2TapCount;
    if (count == 0) { t2_active_extent = 0.0f; return; }

    /* Active tail extent for the global recirc (relative-to-loudest threshold so
     * it tracks density-fade and the decay envelope, both folded into effGains). */
    int32_t maxg = 0;
    for (int t = 0; t < count; t++) {
        int32_t g = effGains_t2[t]; if (g < 0) g = -g;
        if (g > maxg) maxg = g;
    }
    int32_t thr = maxg / 20;   /* 5% of the loudest tap */
    uint32_t ext = 0;
    for (int t = 0; t < count; t++) {
        int32_t g = effGains_t2[t]; if (g < 0) g = -g;
        if (g > thr && (uint32_t)effOffT2L[t] > ext) ext = (uint32_t)effOffT2L[t];
    }
    t2_active_extent = (float)ext;

    /* === L pass === */
    dma_buf_idx = 0;
    dma2_fetch_tap((uint32_t)effOffT2L[0], dma_scratch[0]);
    for (int t = 0; t < count; t++) {
        dma2_wait();
        uint8_t cur = dma_buf_idx;
        if (t + 1 < count) {
            dma_buf_idx ^= 1;
            dma2_fetch_tap((uint32_t)effOffT2L[t + 1], dma_scratch[dma_buf_idx]);
        }

        uint32_t gain = (uint32_t)(int32_t)effGains_t2[t];
        const int16_t *buf = dma_scratch[cur];
        for (int i = 0; i < REVERB_BLOCK; i += 2) {
            uint32_t s_pair = *((const u32_alias *)(buf + i));
            int32_t p0 = smulbb(s_pair, gain);
            int32_t p1 = smultb(s_pair, gain);
            accL[i]   = qadd_sat(accL[i],   p0);
            accL[i+1] = qadd_sat(accL[i+1], p1);
        }
    }

    /* === R pass === */
    dma_buf_idx = 0;
    dma2_fetch_tap((uint32_t)effOffT2R[0], dma_scratch[0]);
    for (int t = 0; t < count; t++) {
        dma2_wait();
        uint8_t cur = dma_buf_idx;
        if (t + 1 < count) {
            dma_buf_idx ^= 1;
            dma2_fetch_tap((uint32_t)effOffT2R[t + 1], dma_scratch[dma_buf_idx]);
        }

        uint32_t gain = (uint32_t)(int32_t)effGains_t2[t];
        const int16_t *buf = dma_scratch[cur];
        for (int i = 0; i < REVERB_BLOCK; i += 2) {
            uint32_t s_pair = *((const u32_alias *)(buf + i));
            int32_t p0 = smulbb(s_pair, gain);
            int32_t p1 = smultb(s_pair, gain);
            accR[i]   = qadd_sat(accR[i],   p0);
            accR[i+1] = qadd_sat(accR[i+1], p1);
        }
    }
}

/* Pre-T2 tanh saturation. Applied to the freshly-written block of t2_ring
 * (Tm bridge output + this block's T2 recirc feedback) so all future T2 tap
 * reads see the saturated signal. Mirrors the JS prototype's "Pre T2"
 * insertion point. Pregain drives the tanh; postgain trims the result.
 * Values are hardcoded to the user's chosen 2.0 / 0.5; expose as globals
 * later if they need to tweak. */
#define SAT_PRE_T2_PREGAIN   1.1f
#define SAT_PRE_T2_POSTGAIN  1.05f

/* Padé(3,2) approximation of tanh: x(27 + x²) / (27 + 9x²).
 * Accurate to <0.001 for |x| ≤ 2 (well below audible). Asymptotes to ±1/3 ×
 * sign(x) → 0.333 instead of ±1 for huge x, but inputs here are bounded
 * within [-2, 2] by SAT_PRE_T2_PREGAIN × (int16/32768) so it never matters.
 * Cost: 4 muls + 1 div + 2 adds ≈ 8 FPU cycles vs ~50 cycles for libm tanhf. */
static inline float fast_tanh(float x)
{
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static void apply_pre_t2_sat(void)
{
    const float scale_in  = 1.0f / 32768.0f;
    const float scale_out = 32768.0f;
    /* Hoist the pregain×scale_in product into a single multiply per sample.
     * The /HEADROOM on in_gain and ×HEADROOM on out_gain cancel the internal
     * headroom scaling for this stage only, so the tanh sees the same drive and
     * the saturation character is unchanged while the ring stays cooled. */
    const float in_gain  = SAT_PRE_T2_PREGAIN * scale_in * REVERB_OUT_MAKEUP;
    const float out_gain = SAT_PRE_T2_POSTGAIN * scale_out * REVERB_HEADROOM;
    for (int n = 0; n < REVERB_BLOCK; n++) {
        uint32_t wi = (block_write_idx + (uint32_t)n) & T2_RING_MASK;
        float v = (float)t2_ring[wi] * in_gain;
        t2_ring[wi] = clamp_f_to_int16(fast_tanh(v) * out_gain);
    }
}

static void do_t2_recirc(void)
{
    float amt = reverb_t2_recirc;
    if (amt > RECIRC_AMOUNT_CLAMP) amt = RECIRC_AMOUNT_CLAMP;
    if (amt <= 0.0f) return;

    const int base_idx = T0_RECIRC_TAPS + T1_RECIRC_TAPS;
    float wSum = t2_recirc_effGain[0] + t2_recirc_effGain[1] + t2_recirc_effGain[2];
    if (wSum < 1e-6f) return;
    float invW = 1.0f / wSum;

    /* Per-tap state. iBase / frac as before; gn folds g × invW so the
     * per-sample loop drops one multiply, and active[] hoists the `g <= 0`
     * branch out of the hot path. */
    int32_t iBase [T2_RECIRC_TAPS];
    float   frac  [T2_RECIRC_TAPS];
    float   gn    [T2_RECIRC_TAPS];
    int     active[T2_RECIRC_TAPS];
    float   prev_a[T2_RECIRC_TAPS];  /* t2_ring[iBase[r] + n]; rolls as n advances */
    int     nActive = 0;
    for (int r = 0; r < T2_RECIRC_TAPS; r++) {
        float g = t2_recirc_effGain[r];
        if (g <= 0.0f) { active[r] = 0; continue; }
        active[r] = 1;
        nActive++;
        gn[r] = g * invW;

        int pidx = base_idx + r;
        recirc_mod_phases[pidx] += recirc_mod_rates[pidx] * (float)REVERB_BLOCK;
        if (recirc_mod_phases[pidx] >  PI_F) recirc_mod_phases[pidx] -= TWO_PI_F;
        if (recirc_mod_phases[pidx] < -PI_F) recirc_mod_phases[pidx] += TWO_PI_F;
        float modOff = (float)t2_recirc_effOff[r]
                     + t2_rc_mod_depth_samples * triWave(recirc_mod_phases[pidx]);
        float pos0  = (float)block_write_idx - modOff;
        int32_t iP  = (int32_t)floorf(pos0);
        iBase[r]    = iP;
        frac [r]    = pos0 - (float)iP;

        /* Seed prev_a with the n=0 read. The inner loop then only needs the
         * "next" sample (b) per tap per output, since this iteration's b
         * becomes the next iteration's a. Halves SDRAM reads in the inner
         * loop (96 → ~51 per block at full density). */
        uint32_t idx0 = ((uint32_t)iBase[r]) & T2_RING_MASK;
        prev_a[r] = (float)t2_ring[idx0];
    }
    if (nActive == 0) return;

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float rcSum = 0.0f;
        for (int r = 0; r < T2_RECIRC_TAPS; r++) {
            if (!active[r]) continue;
            uint32_t idx1 = ((uint32_t)(iBase[r] + n + 1)) & T2_RING_MASK;
            float a = prev_a[r];
            float b = (float)t2_ring[idx1];
            rcSum += gn[r] * (a + frac[r] * (b - a));
            prev_a[r] = b;
        }
        /* gn already folded invW, so rcSum is already weighted-averaged. */
        recirc_damp_t2 += recirc_damp_coeff * (rcSum - recirc_damp_t2);
        float shelf = recirc_damp_t2 + reverb_recirc_damp_hf_gain * (rcSum - recirc_damp_t2);
        recirc_hp_t2 += recirc_hp_coeff * (shelf - recirc_hp_t2);
        float fb = shelf - recirc_hp_t2;   /* high-shelf damping + 100 Hz HPF */

        uint32_t wi = (block_write_idx + (uint32_t)n) & T2_RING_MASK;
        float current = (float)t2_ring[wi];
        t2_ring[wi] = soft_clip_int16(current + amt * fb);
    }
}

/* ==== Global recirculation (T2 tail -> T0 input) ====
 *
 * Reads GLOBAL_RECIRC_TAPS modulated, interpolated taps from deep in the T2
 * ring (fixed fractions of the current T2 window, so the loop delay tracks the
 * reverb size), one-pole damps the weighted average, and mixes it into the
 * fresh T0 input block. Forms a long T0->T1->T2->T0 feedback loop on top of the
 * per-stage recirc — the global "size"/sustain control.
 *
 * Runs after do_input_write_t0 (t0_ring fresh block written, block_write_idx
 * set) and before do_t0_phase, so T0 convolves the augmented input. t2_ring
 * holds the previous block's tail (T2 for this block runs later in poll), so
 * this is a 1-block-delayed feedback like the per-stage recircs.
 *
 * Both rings run at REVERB_HEADROOM level, so no extra headroom scaling is
 * needed — amt is a pure ratio. Mod uses the recirc LFO slots above the
 * per-stage taps; depth reuses t2_rc_mod_depth_samples (scales with T2 window). */
/* Fractions of the active-tail extent at which the global taps read. Clustered
 * near the END so the loop delay ≈ the tail length (each feedback pass appends
 * one tail-length: smooth extension) rather than mid-tail (overlap → ringing) or
 * past the last tap (echo). */
static const float global_recirc_frac[GLOBAL_RECIRC_TAPS] = { 0.85f, 0.93f, 1.0f };

static void do_global_recirc(void)
{
    float amt = reverb_global_recirc;
    if (amt > RECIRC_AMOUNT_CLAMP) amt = RECIRC_AMOUNT_CLAMP;
    if (amt <= 0.0f) return;

    const int base_idx = T0_RECIRC_TAPS + T1_RECIRC_TAPS + T2_RECIRC_TAPS;
    const float invN = 1.0f / (float)GLOBAL_RECIRC_TAPS;

    /* Loop length ≈ active tail extent (so the 1.0 tap reads at the last active
     * tap, not past it → no echo), floored at the T1 window so a tiny tail can't
     * collapse the loop into an audible comb. Uses last block's extent. */
    float gExtent = (t2_active_extent > t1_window_morph) ? t2_active_extent : t1_window_morph;

    int32_t iBase [GLOBAL_RECIRC_TAPS];
    float   frac  [GLOBAL_RECIRC_TAPS];
    float   prev_a[GLOBAL_RECIRC_TAPS];
    for (int r = 0; r < GLOBAL_RECIRC_TAPS; r++) {
        int pidx = base_idx + r;
        recirc_mod_phases[pidx] += recirc_mod_rates[pidx] * (float)REVERB_BLOCK;
        if (recirc_mod_phases[pidx] >  PI_F) recirc_mod_phases[pidx] -= TWO_PI_F;
        if (recirc_mod_phases[pidx] < -PI_F) recirc_mod_phases[pidx] += TWO_PI_F;
        float baseOff = global_recirc_frac[r] * gExtent;
        float modOff  = baseOff + t2_rc_mod_depth_samples * triWave(recirc_mod_phases[pidx]);
        if (modOff < (float)TAP_MIN_OFFSET) modOff = (float)TAP_MIN_OFFSET;
        float pos0 = (float)block_write_idx - modOff;
        int32_t iP = (int32_t)floorf(pos0);
        iBase[r]   = iP;
        frac [r]   = pos0 - (float)iP;
        prev_a[r]  = (float)t2_ring[((uint32_t)iP) & T2_RING_MASK];
    }

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float rcSum = 0.0f;
        for (int r = 0; r < GLOBAL_RECIRC_TAPS; r++) {
            uint32_t idx1 = ((uint32_t)(iBase[r] + n + 1)) & T2_RING_MASK;
            float a = prev_a[r];
            float b = (float)t2_ring[idx1];
            rcSum += (a + frac[r] * (b - a));
            prev_a[r] = b;
        }
        rcSum *= invN;
        recirc_damp_global += recirc_damp_coeff * (rcSum - recirc_damp_global);
        float shelf = recirc_damp_global + reverb_recirc_damp_hf_gain * (rcSum - recirc_damp_global);
        recirc_hp_global += recirc_hp_coeff * (shelf - recirc_hp_global);
        float fb = shelf - recirc_hp_global;   /* high-shelf damping + 100 Hz HPF */

        uint32_t wi = (block_write_idx + (uint32_t)n) & T0_RING_MASK;
        float current = (float)t0_ring[wi];
        t0_ring[wi] = soft_clip_int16(current + amt * fb);
    }
}

static void do_finalize(void)
{
    /* Biquad coefficient recompute is expensive (~500 cycles per call —
     * sinf + cosf + division). With macros tied to ADC pots, raw `!=` would
     * recompute every block from sample-rate ADC noise. Hysteresis at 0.5 %
     * of the current freq is well below audibility (≪ ¼ semitone) but
     * caps recompute rate to a handful per second during a smooth sweep. */
    float lpf_target = reverb_lpf_hz;
    if (lpf_target < 200.0f) lpf_target = 200.0f;
    if (lpf_target > 11900.0f) lpf_target = 11900.0f;
    if (fabsf(lpf_target - last_lpf_hz) > last_lpf_hz * 0.005f) {
        last_lpf_hz = lpf_target;
        biquad_lpf(&lpf_L, lpf_target, 48000.0f);
        biquad_lpf(&lpf_R, lpf_target, 48000.0f);
    }
    float hpf_target = reverb_hpf_hz;
    if (hpf_target < 20.0f)   hpf_target = 20.0f;
    if (hpf_target > 2000.0f) hpf_target = 2000.0f;
    if (fabsf(hpf_target - last_hpf_hz) > last_hpf_hz * 0.005f) {
        last_hpf_hz = hpf_target;
        biquad_hpf(&hpf_L, hpf_target, 48000.0f);
        biquad_hpf(&hpf_R, hpf_target, 48000.0f);
    }

    uint8_t write_buf = (uint8_t)(output_buffer_id ^ 1);
    int16_t *wL = outL[write_buf];
    int16_t *wR = outR[write_buf];
    float pL = prev_out_L;
    float pR = prev_out_R;

    for (int i = 0; i < REVERB_BLOCK; i++) {
        /* Mid-side stereo width on the reverb-rate stereo pair, applied before
         * the 2x upsample interp so the interpolated sample, the original
         * sample, and the carried-over prev all stay width-consistent. The
         * widened side can exceed int16, but the biquad output clamp handles
         * the final range. */
        /* Make up the internal headroom here (the input was attenuated by
         * REVERB_HEADROOM). soft_saturate_q15 soft-clips the cooled accumulator
         * at full scale first, so the makeup can't reintroduce a hard over. */
        float rawL = (float)soft_saturate_q15(accL[i]) * REVERB_OUT_MAKEUP;
        float rawR = (float)soft_saturate_q15(accR[i]) * REVERB_OUT_MAKEUP;
        float m = 0.5f * (rawL + rawR);
        float s = 0.5f * (rawL - rawR) * REVERB_STEREO_WIDTH;
        float curL = m + s;
        float curR = m - s;

        float midL = (pL + curL) * 0.5f;
        float midR = (pR + curR) * 0.5f;
        wL[2 * i]     = soft_clip_int16(biquad_process(&lpf_L, biquad_process(&hpf_L, midL)));
        wR[2 * i]     = soft_clip_int16(biquad_process(&lpf_R, biquad_process(&hpf_R, midR)));
        wL[2 * i + 1] = soft_clip_int16(biquad_process(&lpf_L, biquad_process(&hpf_L, curL)));
        wR[2 * i + 1] = soft_clip_int16(biquad_process(&lpf_R, biquad_process(&hpf_R, curR)));

        pL = curL;
        pR = curR;
    }
    /* DIAGNOSTIC ABANDONED — click persisted with wR=wL mirror, so it's not
     * inside the velvet reverb's stereo split. Revert leaves R using its
     * own computed output (lpf_R/hpf_R/accR/T2-R-pass path). */
    prev_out_L = pL;
    prev_out_R = pR;

    __DMB();
    output_buffer_id = write_buf;
}

/* ==== ISR-side: input buffer fill ==== */
void velvet_reverb_push_sample(int16_t input)
{
    push_acc += input;
    push_phase ^= 1;
    if (push_phase) return;

    int16_t avg = (int16_t)(push_acc >> 1);
    push_acc = 0;

    input_fill[input_fill_idx] = avg;
    input_fill_idx++;

    if (input_fill_idx >= REVERB_BLOCK) {
        input_fill_idx = 0;
        if (!block_ready) {
            int16_t *tmp = input_fill;
            input_fill = input_ready;
            input_ready = tmp;
            block_ready = 1;
        } else {
#ifndef VELVET_REVERB_HOST
            diag_log(DIAG_EVT_REVERB_DROP, 0);
#endif
        }
    }
}

/* ==== Main-loop: drain one block ==== */
#ifndef REVERB_STAGE
#define REVERB_STAGE 4
#endif

void velvet_reverb_poll(void)
{
    if (!block_ready) return;

    /* DIAGNOSTIC RESULT (poll short-circuit): clicks persisted. Reverb is
     * innocent — bug is in channel-1's looping-delay path. Reverb restored
     * so the compress() bypass test has a realistic audio signal. */

#ifndef VELVET_REVERB_HOST
    uint32_t t0 = DWT->CYCCNT;
#endif

    /* Cheap per-block morph update (IIR + recirc offsets). The heavier
     * effGains recompute is deferred to outside the block timer below. */
#ifndef VELVET_REVERB_HOST
    uint32_t t_morph_0 = DWT->CYCCNT;
#endif
    update_morph_state();
#ifndef VELVET_REVERB_HOST
    diag_log(DIAG_EVT_REVERB_MORPH, DWT->CYCCNT - t_morph_0);
#endif

#if REVERB_STAGE >= 1
    do_input_write_t0();
#endif
#if REVERB_STAGE >= 3
    /* Global T2->T0 feedback, injected into the fresh T0 input before T0 conv.
     * Reads last block's T2 tail (T2 for this block runs further down). */
    do_global_recirc();
#endif
#if REVERB_STAGE >= 2
    {
#ifndef VELVET_REVERB_HOST
        uint32_t t0_t0 = DWT->CYCCNT;
#endif
        do_t0_phase();
#ifndef VELVET_REVERB_HOST
        uint32_t t0_rc_0 = DWT->CYCCNT;
#endif
        do_t0_recirc();
#ifndef VELVET_REVERB_HOST
        uint32_t t0_end = DWT->CYCCNT;
        diag_log(DIAG_EVT_REVERB_T0_RECIRC, t0_end - t0_rc_0);
        diag_log(DIAG_EVT_REVERB_T0,        t0_end - t0_t0);
#endif
        do_bridge_t0_to_t1();
    }
    {
#ifndef VELVET_REVERB_HOST
        uint32_t t1_t0 = DWT->CYCCNT;
#endif
        do_t1_phase();
#ifndef VELVET_REVERB_HOST
        uint32_t t1_rc_0 = DWT->CYCCNT;
#endif
        do_t1_recirc();
#ifndef VELVET_REVERB_HOST
        uint32_t t1_end = DWT->CYCCNT;
        diag_log(DIAG_EVT_REVERB_T1_RECIRC, t1_end - t1_rc_0);
        diag_log(DIAG_EVT_REVERB_T1,        t1_end - t1_t0);
#endif
        do_bridge_t1_to_t2();
    }
#endif
#if REVERB_STAGE >= 3
    {
#ifndef VELVET_REVERB_HOST
        uint32_t t2_t0 = DWT->CYCCNT;
#endif
        do_t2_phase();
#ifndef VELVET_REVERB_HOST
        uint32_t t2_rc_0 = DWT->CYCCNT;
#endif
        do_t2_recirc();
        /* Pre-T2 tanh sat — saturates Tm + recirc content in t2_ring before
         * subsequent block reads it. Runs unconditionally (always-on per
         * user spec). Counts toward T2_RECIRC timing. */
        apply_pre_t2_sat();
#ifndef VELVET_REVERB_HOST
        uint32_t t2_end = DWT->CYCCNT;
        diag_log(DIAG_EVT_REVERB_T2_RECIRC, t2_end - t2_rc_0);
        diag_log(DIAG_EVT_REVERB_T2,        t2_end - t2_t0);
#endif
    }
#endif
#if REVERB_STAGE >= 4
#ifndef VELVET_REVERB_HOST
    uint32_t t_fin_0 = DWT->CYCCNT;
#endif
    do_finalize();
#ifndef VELVET_REVERB_HOST
    diag_log(DIAG_EVT_REVERB_FINALIZE, DWT->CYCCNT - t_fin_0);
#endif
#endif

#ifndef VELVET_REVERB_HOST
    diag_log(DIAG_EVT_REVERB_BLOCK, DWT->CYCCNT - t0);
#endif

    block_ready = 0;

    /* Background update — NOT inside the block timer but timed separately
     * so we can see how much CPU it eats between blocks. Whatever it costs
     * is unavailable to the next block; if it stretches past the input
     * buffer's lead time we miss a deadline even though the per-stage
     * numbers look fine. */
#ifndef VELVET_REVERB_HOST
    uint32_t t_bg_0 = DWT->CYCCNT;
#endif
    background_eff_gains_update();
#ifndef VELVET_REVERB_HOST
    diag_log(DIAG_EVT_REVERB_BG_EFF, DWT->CYCCNT - t_bg_0);
#endif
}

/* ==== Output ====
 * Each channel samples output_buffer_id independently when its idx wraps.
 * No shared state, no order dependence between the two codec ISRs. */
int16_t velvet_reverb_out_left(void)
{
    if (out_idx_L == 0)
        reader_buf_L = output_buffer_id;
    int16_t s = outL[reader_buf_L][out_idx_L];
    out_idx_L = (uint8_t)((out_idx_L + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}

int16_t velvet_reverb_out_right(void)
{
    if (out_idx_R == 0)
        reader_buf_R = output_buffer_id;
    int16_t s = outR[reader_buf_R][out_idx_R];
    out_idx_R = (uint8_t)((out_idx_R + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}
