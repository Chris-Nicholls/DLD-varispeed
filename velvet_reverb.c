/*
 * velvet_reverb.c — Velvet sparse-noise convolution reverb for DLD
 *
 * Three-stage cascade (T0 → T1 → T2) at half codec rate (24 kHz), processed
 * from the main loop in 16-sample blocks (= 32 codec samples, ~667 µs).
 *
 *   codec → push (avg 2:1 decimate) → pre-delay sustain engine (2 modulated,
 *           damped feedback delay lines A/B, shared shelves, input duck, mix)
 *             ↓ (feedforward into the cascade)
 *           T0 ring (CCM)
 *           T0 sparse conv  (+ optional per-tap Lexicon LFO on the readout)
 *             ↓ saturate
 *           T1 ring (CCM)
 *           T1 sparse conv
 *             ↓ saturate
 *           T2 ring (SDRAM, DMA-prefetched)
 *           T2 sparse conv (stereo via independent L/R offsets, shared gain)
 *             ↓ saturate + 2× linear-interp upsample
 *           HPF + LPF biquads @ 48 kHz (LPF doubles as reconstruction filter)
 *             ↓
 *           codec output
 *
 * Dynamic-parameter infrastructure:
 *
 *   - Density is FIXED at MAX: every stage generates MAX taps once at init
 *     (sign + 1/√N; T0 keeps the cosine tail-taper). There is no runtime
 *     density control and no per-tap removal rank.
 *   - Window sizes are mutable globals smoothed per-block with a one-pole IIR
 *     (driven by the Decay macro). Each tap has a relocation ladder so it
 *     migrates one-at-a-time as the window shrinks.
 *   - Each block we recompute the *effective* per-tap gain (int16 Q15):
 *         base × window-fade × [T2 exp envelope] × gain-comp
 *     and the convolution inner loops read from these effGains arrays, so
 *     window/envelope/loudness all change smoothly without regenerating taps.
 *   - The effGains recompute is round-robin across the three stages (one
 *     per block) to keep the per-block update cost flat.
 *   - The long tail / sustain comes from the pre-delay feedback lines in
 *     front of the cascade (the cascade itself is purely feedforward), which
 *     replaces the old per-stage + global recirculation.
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
/* Dual 16x16 multiply-accumulate: acc + a.lo*b.lo + a.hi*b.hi (no saturation). */
__attribute__((always_inline)) static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    return acc + (int32_t)(int16_t)(a & 0xFFFFu) * (int32_t)(int16_t)(b & 0xFFFFu)
               + (int32_t)(int16_t)(a >> 16)     * (int32_t)(int16_t)(b >> 16);
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
/* Dual 16x16 multiply-accumulate: acc + a.lo*b.lo + a.hi*b.hi (single cycle). */
__attribute__((always_inline)) static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm volatile ("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
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
/* Wrap modulus for ring_write_idx/block_write_idx (see do_input_write_t0). 2^23 is
 * a multiple of every ring size (T0=2^13, T1=2^14, T2=2^18) so masked positions are
 * continuous across the wrap, and < 2^24 so the float cast in the recirc math is exact. */
#define REVERB_IDX_WRAP  0x800000UL   /* 2^23 = 8388608 */

/* ==== Tap arrays (generated once at MAX count + MAX window) ====
 * Density is fixed at MAX, so the per-tap removal-rank arrays are gone. */
uint32_t t0TapOffsets[MAX_T0_TAPS];
int16_t  t0TapGains [MAX_T0_TAPS];
int      t0TapCount = 0;

uint32_t t1TapOffsets[MAX_T1_TAPS];
int16_t  t1TapGains [MAX_T1_TAPS];
int      t1TapCount = 0;

uint32_t t2TapOffsetsL[MAX_T2_TAPS];
uint32_t t2TapOffsetsR[MAX_T2_TAPS];
int16_t  t2TapGains   [MAX_T2_TAPS];
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
#ifdef VELVET_REVERB_HOST
static float host_effwin_t2[MAX_T2_TAPS];  /* relocation/window gain only (no decay env) */
#endif

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
/* Defaults below match the user's velvet_buffers slider snapshot. Macro-driven
 * params (windows, feedback, mix, lpf/hpf, high-shelf Hz) are recomputed from
 * the Decay/Tone pots at init. */
float reverb_lpf_hz    = 8600.0f;   /* Tone → MP_LPF */
float reverb_hpf_hz    = 170.0f;    /* Tone → MP_HPF */

float reverb_t0_window_ms_target  = 69.0f;     /* Decay → JS paramT1Window */
float reverb_t1_window_ms_target  = 536.0f;    /* Decay → JS paramTmWindow */
float reverb_t2_duration_s_target = 3.217f;    /* Decay → JS paramT2Duration */

/* ==== Pre-delay sustain engine params ==== Macro-driven: feedback, delay_mix,
 * high-shelf Hz. Fixed (no macro maps them): delay times, low-shelf, high-shelf
 * dB, fb-mod, duck. All values from the user's slider snapshot. */
float reverb_feedback         = 0.969f;  /* Decay → JS paramFeedback */
float reverb_predelay_a_s      = 0.185f;  /* fixed: JS reverbDelayTimeA */
float reverb_predelay_b_s      = 0.300f;  /* fixed: JS reverbDelayTimeB */
float reverb_delay_mix         = 0.68f;   /* Decay → JS reverbDelayMix */
float reverb_fb_low_shelf_hz   = 200.0f;  /* fixed: JS feedbackLowShelfFreq */
float reverb_fb_low_shelf_db   = -2.5f;   /* fixed: JS feedbackLowShelfDb */
float reverb_fb_high_shelf_hz  = 2000.0f; /* Tone → JS feedbackHighShelfFreq */
float reverb_fb_high_shelf_db  = -0.5f;   /* fixed: JS feedbackHighShelfDb */
float reverb_fb_mod_depth      = 0.01f;   /* fixed: JS feedbackModDepth */
float reverb_fb_mod_rate       = 0.35f;   /* fixed: JS feedbackModRate */
float reverb_duck_amount       = 1.0f;    /* fixed: JS delayDuckAmount */
float reverb_duck_release_s    = 2.0f;    /* fixed: JS delayDuckRelease */

/* T0 per-tap Lexicon LFO (early-reflection smearer; replaces recirc mod).
 * Fixed defaults from JS paramT1TapMod* (JS T1 → C T0). */
float reverb_t0_tap_mod_depth  = 0.26f;
float reverb_t0_tap_mod_rate   = 0.59f;

#define TAP_COMP_CLAMP        2.0f       /* max gain-comp boost (avoid int16 overflow) */

/* ~12 Hz DC-block one-pole in each pre-delay feedback loop (JS PRE_HP_COEFF). */
#define PRE_HP_FREQ_HZ        12.0f
/* Fixed ~2 ms duck attack (JS DUCK_ATK_COEF). */
#define DUCK_ATK_SEC          0.002f
/* Minimum pre-delay read time (JS preMinT = 5 ms) so reads stay in the past. */
#define PRE_MIN_TIME_SAMPLES  (0.005f * (float)REVERB_FS_HZ)
/* depth = 1 → ~5 ms T0 per-tap swing (JS T1_TAP_MOD_MAX). */
#define T0_TAP_MOD_MAX_SAMPLES  (0.005f * (float)REVERB_FS_HZ)
#define GOLDEN_ANGLE_RAD      2.39996322972865332f

/* Mid-side stereo width applied to the final T2 output in do_finalize.
 * 1.0 = neutral, <1 narrows toward mono, >1 widens by boosting the
 * decorrelated (L-R) component. Only T2 is stereo (T0/T1 are mono), so this
 * directly scales that decorrelation. Hard-coded for now (JS paramStereoWidth). */
#define REVERB_STEREO_WIDTH   1.47f

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

/* ==== Macro system ====
 * Two macros (Decay, Tone), each 0..1. Every mapped param uses the full macro
 * range — macro value u lerps bound_min..bound_max (exp where both > 0). */
typedef enum {
    MP_T0_WINDOW,
    MP_T1_WINDOW,
    MP_T2_DURATION,
    MP_FEEDBACK,
    MP_DELAY_MIX,
    MP_LPF,
    MP_HPF,
    MP_FB_HIGH_SHELF_HZ,
    MP_COUNT
} macro_param_id_t;

typedef struct {
    float *target;
    float bound_min;
    float bound_max;
    uint8_t use_exp;       /* 1 = exp lerp when bounds both > 0; else linear */
} macro_param_t;

typedef struct {
    float value;
    int   num_params;
    const macro_param_id_t *params;
} macro_t;

/* Bounds + use_exp mirror velvet_param_bounds. Stage remap JS→firmware:
 * JS paramT1→C T0, JS paramTm→C T1, JS paramT2→C T2. Unmapped globals
 * (predelay times, shelf dB/freq-low, duck, fb-mod) keep fixed defaults. */
static macro_param_t reverb_macro_params[MP_COUNT] = {
    [MP_T0_WINDOW]        = { &reverb_t0_window_ms_target,   28.0f,    80.0f,  1 },
    [MP_T1_WINDOW]        = { &reverb_t1_window_ms_target,  150.0f,   660.0f,  1 },
    [MP_T2_DURATION]      = { &reverb_t2_duration_s_target,   0.8f,     4.0f,  1 },
    [MP_FEEDBACK]         = { &reverb_feedback,               0.8f,     1.0f,  1 },
    [MP_DELAY_MIX]        = { &reverb_delay_mix,             -1.3f,     1.0f,  0 }, /* linear: neg bound */
    [MP_LPF]              = { &reverb_lpf_hz,               500.0f, 20000.0f,  1 },
    [MP_HPF]              = { &reverb_hpf_hz,               100.0f,   200.0f,  1 },
    [MP_FB_HIGH_SHELF_HZ] = { &reverb_fb_high_shelf_hz,    1000.0f,  2500.0f,  1 },
};

static const macro_param_id_t decay_params[] = {
    MP_T2_DURATION, MP_T0_WINDOW, MP_T1_WINDOW, MP_FEEDBACK, MP_DELAY_MIX,
};
static const macro_param_id_t tone_params[] = {
    MP_LPF, MP_HPF, MP_FB_HIGH_SHELF_HZ,
};

#define M_COUNT(arr) (int)(sizeof(arr) / sizeof((arr)[0]))
#define MACRO_DECAY  0
#define MACRO_TONE   1
#define NUM_MACROS   2
static macro_t reverb_macros[NUM_MACROS] = {
    { 0.86f, M_COUNT(decay_params), decay_params },
    { 0.77f, M_COUNT(tone_params),  tone_params  },
};

/* Macro recompute is ~14 powf calls (~17 µs). Called from params.c at
 * main-loop rate (~1 kHz), but the morph IIRs downstream have a 30 ms tau,
 * so anything faster than ~100 Hz is wasted work. apply_*_macro now just
 * stores the value + sets a dirty flag; the actual recompute happens at
 * audio-block rate, throttled. */
static volatile uint8_t macros_dirty = 1;   /* force initial recompute */

void velvet_reverb_recompute_macros(void)
{
    for (int m = 0; m < NUM_MACROS; m++) {
        float u = reverb_macros[m].value;
        if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
        const macro_param_id_t *pids = reverb_macros[m].params;
        int n = reverb_macros[m].num_params;
        for (int i = 0; i < n; i++) {
            const macro_param_t *mp = &reverb_macro_params[pids[i]];
            float val;
            if (mp->use_exp && mp->bound_min > 0.0f && mp->bound_max > 0.0f &&
                mp->bound_min != mp->bound_max) {
                val = mp->bound_min * powf(mp->bound_max / mp->bound_min, u);
            } else {
                val = mp->bound_min + u * (mp->bound_max - mp->bound_min);
            }
            *mp->target = val;
        }
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

/* The Decay→feedback curve (bounds {0.8, 1.0}, exp) is extremely sensitive at
 * the top: the last few % of travel cover fb 0.994→0.999, which is roughly a
 * 6x tail-length change and a ~9 dB drop in the sustain plateau. Hardware pots
 * rarely reach full ADC scale, so the pre-delay tail would never reach the
 * full (JS-matching) length the app gets at macro=100. Saturate the top of the
 * Decay macro so the last DECAY_MACRO_TOP_SAT of travel maps to full — this
 * leaves the true-full behaviour identical to JS, it only makes it reachable. */
#define DECAY_MACRO_TOP_SAT  0.96f
void velvet_reverb_apply_decay_macro  (float v) {
    v *= (1.0f / DECAY_MACRO_TOP_SAT);          /* set_macro_value clamps to [0,1] */
    set_macro_value(MACRO_DECAY, v);
}
void velvet_reverb_apply_tone_macro   (float v) { set_macro_value(MACRO_TONE, v); }

/* ==== Morph alphas (per block, derived from per-tau / blocks-per-sec) ====
 * Block rate = fs_reverb / REVERB_BLOCK = 24000 / 16 = 1500 Hz.
 *   tau ≈ 1/(α × 1500). 0.0219 → ~30 ms (window), 0.00664 → ~100 ms (decay). */
#define ALPHA_WINDOW_MORPH   0.0219f
/* ==== Morph state ====
 * Density is fixed at MAX, so there is no tap-count morph. Windows morph
 * (driven by the Decay macro); gain-comp morph tracks the resulting energy. */
static float t0_window_morph;       /* samples at fs_reverb */
static float t1_window_morph;
static float t2_window_morph;
static float t0_tap_comp_morph;
static float t1_tap_comp_morph;
static float t2_tap_comp_morph;
static float t0_full_energy;        /* sum(baseGain²) — set at IR-gen */
static float t1_full_energy;
static float t2_full_energy;

/* Pre-delay delay times are smoothed (avoids zipper / pitch-glide when the
 * Decay macro sweeps the loop length); feedback/mix/shelf changes are gain-
 * like and used directly. Samples at fs_reverb. */
static float predelay_a_morph;
static float predelay_b_morph;

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
/* Smoothed Tone-macro cutoffs. The macro writes reverb_lpf_hz/reverb_hpf_hz/
 * reverb_fb_high_shelf_hz directly (in ADC-sized steps), so feeding them to
 * the biquads raw produced a staircase of coefficient jumps = zipper on a
 * Tone sweep. These IIR-track the targets (per reverb block, ~30 ms tau, same
 * as the window morphs) so the biquads see a continuous cutoff. */
static float lpf_hz_morph   = 8600.0f;
static float hpf_hz_morph   = 170.0f;
static float fb_hs_hz_morph = 2000.0f;

/* ==== Pre-delay sustain engine state ====
 * Two float feedback delay lines in SDRAM (see velvet_reverb.h). Shared loop
 * gain + shelves (tone/damping), per-line modulation + DC block, input duck,
 * and a dry/wet crossfade into the cascade. Mirrors JS velvet_kernel lines
 * 599-663. The line storage is float so the high-feedback loop doesn't
 * accumulate int16 quantisation noise. */
static float * const predelay_a = (float *)PRE_DELAY_A_BASE;
static float * const predelay_b = (float *)PRE_DELAY_B_BASE;
static uint32_t predelay_wh = 0;            /* shared free-running write head */
static float pre_mod_phase_a = 0.0f;
static float pre_mod_phase_b = PI_F;        /* decorrelated phase */
static float pre_hp_a = 0.0f, pre_hp_b = 0.0f;   /* per-line ~12 Hz DC block state */
static float duck_env = 0.0f;
static Biquad pre_low_shelf_a CCM_ATTR, pre_high_shelf_a CCM_ATTR;
static Biquad pre_low_shelf_b CCM_ATTR, pre_high_shelf_b CCM_ATTR;
/* Cached shelf coeff inputs — recompute only on change. */
static float last_fb_ls_hz = -1.0f, last_fb_ls_db = -999.0f;
static float last_fb_hs_hz = -1.0f, last_fb_hs_db = -999.0f;
static float pre_hp_coeff = 0.0f;
static float duck_atk_coeff = 0.0f;

/* Per-block float scratch holding the (pre-delayed) cascade input, normalised
 * to ±1 (may exceed during sustain build-up; clamped on the int16 ring write
 * in do_input_write_t0). */
static float predelay_out[REVERB_BLOCK] CCM_ATTR;

/* ==== T0 per-tap Lexicon LFO state ====
 * Each early-reflection tap is displaced by a per-tap (golden-angle phase)
 * LFO so the early cluster shimmers diffusely. cos/sin tables seeded at
 * regen; the shared phase advances per block. */
static float t0_tap_mod_phase = 0.0f;
static float t0TapModCos[MAX_T0_TAPS] CCM_ATTR;
static float t0TapModSin[MAX_T0_TAPS] CCM_ATTR;

#ifndef VELVET_REVERB_HOST
static inline void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ==== Stage timing (pure compute, ISR time removed) ====
 *
 * A stage's cost is wall time minus the codec-ISR time that preempted it. Taking
 * that as two independent reads is racy: if an ISR lands BETWEEN reading CYCCNT
 * and reading diag_isr_cycles, one delta contains that ISR and the other does
 * not. Depending on which end it happens at, the result is either inflated or
 * driven negative — and being unsigned, negative wraps to ~4e9 and saturates at
 * DIAG_CYCLES_MASK, i.e. a fake 6241.5 us. That is not a subtle skew: it
 * manufactures the largest possible outlier out of nothing, in exactly the p99
 * column used to hunt overruns, which is worse than having no measurement.
 *
 * So sample the pair atomically, and clamp instead of wrapping if the two still
 * disagree. Interrupts are masked for two register reads only. */
static inline uint32_t diag_stage_start(uint32_t *isr_out)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT->CYCCNT;
    *isr_out = diag_isr_cycles;
    if (!primask) __enable_irq();
    return c;
}

static inline uint32_t diag_stage_cycles(uint32_t cyc0, uint32_t isr0)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT->CYCCNT;
    uint32_t i = diag_isr_cycles;
    if (!primask) __enable_irq();
    uint32_t wall = c - cyc0;
    uint32_t isr  = i - isr0;
    return (wall > isr) ? (wall - isr) : 0u;
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

/* ==== Output-underrun (bitcrush) instrumentation ====
 * blocks_produced is the monotonic count of fully-written output blocks; the
 * output ring readers use it as the publish sequence (see below). It also
 * doubles as the ground-truth defect counter: output_miss_count is bumped only
 * when a reader genuinely underruns the ring (poll produced nothing for two
 * consecutive wraps), independent of the FSK diagnostic load.
 * block_ready_cyc timestamps when an input block goes ready so poll can report
 * how long it sat before the main loop picked it up. */
static volatile uint32_t blocks_produced   = 0;
static volatile uint32_t output_miss_count = 0;
#ifndef VELVET_REVERB_HOST
static volatile uint32_t block_ready_cyc   = 0;
#endif

static int32_t accT0[REVERB_BLOCK] CCM_ATTR;
static int32_t accT1[REVERB_BLOCK] CCM_ATTR;
static int32_t accL [REVERB_BLOCK] CCM_ATTR;
static int32_t accR [REVERB_BLOCK] CCM_ATTR;

/* ==== Output ring (jitter buffer) ====
 * poll() produces one output block per input block (avg rate matches the
 * codec consume rate — reverb_drop is 0). The defect was timing jitter: poll
 * runs asynchronously in the main loop and occasionally finishes a block just
 * *after* the codec reader has already wrapped, so the reader replayed the
 * previous block (audible bitcrush). A 2-deep double buffer had zero slack to
 * absorb that.
 *
 * This ring decouples production from consumption. The writer (poll) fills
 * slot (blocks_produced % NBUF) then publishes by incrementing blocks_produced.
 * Each codec reader advances its own monotonic read sequence by one block per
 * wrap and holds a steady lag of ~1 completed block behind the newest, so a
 * single late/slow poll no longer starves it — only two consecutive misses do.
 * The reader holds a steady TARGET_LAG-block cushion behind the newest
 * published block: it accumulates slack during fast polls and spends it during
 * slow ones, so a run of late/slow polls no longer starves it. Only when the
 * cushion is fully drained (TARGET_LAG consecutive misses) does a real bitcrush
 * occur. NBUF > TARGET_LAG keeps a guard slot so the writer can never overwrite
 * a block being read. Costs TARGET_LAG blocks of latency (~2.7 ms — inaudible
 * on a reverb tail). */
#define REVERB_OUT_NBUF        8
#define REVERB_OUT_TARGET_LAG  4
static int16_t outL[REVERB_OUT_NBUF][REVERB_OUT_BLOCK] CCM_ATTR;
static int16_t outR[REVERB_OUT_NBUF][REVERB_OUT_BLOCK] CCM_ATTR;
/* Per-channel read sequence + cached slot. The two codec ISRs wrap
 * independently (their DMAs can drift), so each tracks the writer on its own;
 * neither depends on the other firing in lockstep. */
static uint32_t rd_seq_L = 0;
static uint32_t rd_seq_R = 0;
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

/* ==== Smoothstep helper for fade curves ==== */
static inline float smoothstep01(float x)
{
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}
/* Per-tap T2 exponential decay envelope (frac ∈ [0,1] = effOff / window).
 * exp falls to ~1e-4 (-80 dB) across the window. T0 has taper baked in;
 * T1 has no decay envelope. Uses exp_env_lut (filled in velvet_reverb_init). */
#define ENV_FRAC_BINS  32
static float exp_env_lut[ENV_FRAC_BINS] CCM_ATTR;

static inline float exp_decay_from_frac(float frac)
{
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    float bin_f = frac * (float)(ENV_FRAC_BINS - 1);
    int i = (int)bin_f;
    if (i > ENV_FRAC_BINS - 2) i = ENV_FRAC_BINS - 2;
    float t = bin_f - (float)i;
    return exp_env_lut[i] + t * (exp_env_lut[i + 1] - exp_env_lut[i]);
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
/* RBJ shelving filters (S = 1 slope). Used inside the pre-delay feedback loop
 * at the reverb rate. dB > 0 boosts, < 0 cuts. Mirrors JS biquadLowShelf /
 * biquadHighShelf. */
static void biquad_lowshelf(Biquad *bq, float freq, float dB, float fs)
{
    float A = powf(10.0f, dB / 40.0f);
    float w0 = TWO_PI_F * freq / fs;
    float cosw = cosf(w0);
    float alpha = sinf(w0) * 0.5f * 1.41421356f;     /* sqrt(2) */
    float tsa = 2.0f * sqrtf(A) * alpha;
    float a0 = (A + 1.0f) + (A - 1.0f) * cosw + tsa;
    bq->b0 =  A * ((A + 1.0f) - (A - 1.0f) * cosw + tsa) / a0;
    bq->b1 =  2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosw) / a0;
    bq->b2 =  A * ((A + 1.0f) - (A - 1.0f) * cosw - tsa) / a0;
    bq->a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosw) / a0;
    bq->a2 =  ((A + 1.0f) + (A - 1.0f) * cosw - tsa) / a0;
}
static void biquad_highshelf(Biquad *bq, float freq, float dB, float fs)
{
    float A = powf(10.0f, dB / 40.0f);
    float w0 = TWO_PI_F * freq / fs;
    float cosw = cosf(w0);
    float alpha = sinf(w0) * 0.5f * 1.41421356f;
    float tsa = 2.0f * sqrtf(A) * alpha;
    float a0 = (A + 1.0f) - (A - 1.0f) * cosw + tsa;
    bq->b0 =  A * ((A + 1.0f) + (A - 1.0f) * cosw + tsa) / a0;
    bq->b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosw) / a0;
    bq->b2 =  A * ((A + 1.0f) + (A - 1.0f) * cosw - tsa) / a0;
    bq->a1 =  2.0f * ((A - 1.0f) - (A + 1.0f) * cosw) / a0;
    bq->a2 =  ((A + 1.0f) - (A - 1.0f) * cosw - tsa) / a0;
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
 * expect slightly higher reverb_T2 timing but well under the deadline.
 *
 * Overridable from the Makefile (`make PROFILE=1 NODMA=1`) so the DMA and
 * CPU-copy paths can be A/B'd by comparing reverb_T2 between two builds. */
#ifndef DMA2_DISABLED_FOR_DIAGNOSTIC
#define DMA2_DISABLED_FOR_DIAGNOSTIC  0   /* revert: not the cause */
#endif

/* Profile-build accumulators, summed across all 64 tap fetches of one block
 * (32 taps x L/R) and emitted by do_t2_phase. Reading DWT->CYCCNT four times
 * per tap costs roughly 1k cycles a block, so a profile build's reverb_T2 is
 * inflated by ~8% relative to a normal build — compare T2 totals only between
 * builds of the same kind. The wait/kick split is unaffected. */
#ifdef DIAG_REVERB_PROFILE
static uint32_t t2_dma_wait_cyc = 0;
static uint32_t t2_dma_kick_cyc = 0;
#endif

static inline void dma2_kick(const int16_t *src, int16_t *dst, uint32_t count)
{
#ifdef DIAG_REVERB_PROFILE
    uint32_t _k0 = DWT->CYCCNT;
#endif
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
#ifdef DIAG_REVERB_PROFILE
    t2_dma_kick_cyc += DWT->CYCCNT - _k0;
#endif
}
static inline void dma2_wait(void)
{
#ifdef DIAG_REVERB_PROFILE
    uint32_t _w0 = DWT->CYCCNT;
#endif
#if DMA2_DISABLED_FOR_DIAGNOSTIC
    /* memcpy is synchronous — nothing to wait for. */
#else
    while ((DMA2_Stream1->CR & DMA_SxCR_EN) && !(DMA2->LISR & DMA_LISR_TCIF1)) {}
    __DMB();
#endif
#ifdef DIAG_REVERB_PROFILE
    t2_dma_wait_cyc += DWT->CYCCNT - _w0;
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
#define MAX_STAGE_TAPS   64   /* working-array bound for build_gapfill_ladders */

/* Density is fixed at MAX, so the recirc geometric ladders and the
 * neighbour-spacing removal-rank computation (build_tap_ladder /
 * build_tap_ladder_lr / compute_tap_ranks) have been removed — only the
 * window relocation gap-fill ladder (build_gapfill_ladders, below) remains. */

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
static void generate_mono_stage(uint32_t *outOffsets, int16_t *outGains,
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
 * with a shared jitter draw, so L/R divergence stays bounded. */
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
    generate_mono_stage(t0TapOffsets, t0TapGains, &t0TapCount,
                        MAX_T0_TAPS, T0_WINDOW_MAX_SAMPLES, /*apply_taper=*/1);

    /* T1 — middle. No taper; decay envelope applied dynamically per-block. */
    generate_mono_stage(t1TapOffsets, t1TapGains, &t1TapCount,
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

    /* T0 per-tap Lexicon-LFO phase table: each tap gets a fixed golden-angle
     * decorrelated phase so the early cluster shimmers diffusely. */
    for (int k = 0; k < MAX_T0_TAPS; k++) {
        float ph = (float)k * GOLDEN_ANGLE_RAD;
        t0TapModCos[k] = cosf(ph);
        t0TapModSin[k] = sinf(ph);
    }
}

/* Per-stage fade-zone minimum samples — keeps the relocation bell smooth even
 * at very short window settings. Matches the JS prototype's per-stage floors
 * (JS T1→C T0 = 192, JS Tm→C T1 = 192, JS T2→C T2 = 512). */
#define T0_FADE_MIN  192.0f
#define T1_FADE_MIN  192.0f
#define T2_FADE_MIN  512.0f
/* Relocation crossfade width as a fraction of the window (JS = 0.06). */
#define FADE_WINDOW_FRAC  0.06f

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
    for (int b = 0; b < REVERB_OUT_NBUF; b++) {
        for (int i = 0; i < REVERB_OUT_BLOCK; i++) {
            outL[b][i] = 0; outR[b][i] = 0;
        }
    }
    prev_out_L = 0; prev_out_R = 0;
    push_acc = 0; push_phase = 0;
    blocks_produced = 0;
    rd_seq_L = 0; rd_seq_R = 0;
    reader_buf_L = 0; reader_buf_R = 0;
    out_idx_L = 0; out_idx_R = 0;

    /* Biquads */
    hpf_L.z1 = hpf_L.z2 = 0.0f;  hpf_R.z1 = hpf_R.z2 = 0.0f;
    lpf_L.z1 = lpf_L.z2 = 0.0f;  lpf_R.z1 = lpf_R.z2 = 0.0f;
    biquad_hpf(&hpf_L, reverb_hpf_hz, 48000.0f);
    biquad_hpf(&hpf_R, reverb_hpf_hz, 48000.0f);
    biquad_lpf(&lpf_L, reverb_lpf_hz, 48000.0f);
    biquad_lpf(&lpf_R, reverb_lpf_hz, 48000.0f);
    last_hpf_hz = reverb_hpf_hz;
    last_lpf_hz = reverb_lpf_hz;
    lpf_hz_morph = reverb_lpf_hz;
    hpf_hz_morph = reverb_hpf_hz;

    /* ---- Pre-delay sustain engine ---- */
    for (uint32_t i = 0; i < PRE_DELAY_LINE_SAMPLES; i++) { predelay_a[i] = 0.0f; predelay_b[i] = 0.0f; }
    predelay_wh = 0;
    pre_mod_phase_a = 0.0f; pre_mod_phase_b = PI_F;
    pre_hp_a = pre_hp_b = 0.0f;
    duck_env = 0.0f;
    pre_low_shelf_a.z1 = pre_low_shelf_a.z2 = 0.0f;
    pre_high_shelf_a.z1 = pre_high_shelf_a.z2 = 0.0f;
    pre_low_shelf_b.z1 = pre_low_shelf_b.z2 = 0.0f;
    pre_high_shelf_b.z1 = pre_high_shelf_b.z2 = 0.0f;
    biquad_lowshelf(&pre_low_shelf_a,  reverb_fb_low_shelf_hz,  reverb_fb_low_shelf_db,  (float)REVERB_FS_HZ);
    biquad_lowshelf(&pre_low_shelf_b,  reverb_fb_low_shelf_hz,  reverb_fb_low_shelf_db,  (float)REVERB_FS_HZ);
    biquad_highshelf(&pre_high_shelf_a, reverb_fb_high_shelf_hz, reverb_fb_high_shelf_db, (float)REVERB_FS_HZ);
    biquad_highshelf(&pre_high_shelf_b, reverb_fb_high_shelf_hz, reverb_fb_high_shelf_db, (float)REVERB_FS_HZ);
    last_fb_ls_hz = reverb_fb_low_shelf_hz;  last_fb_ls_db = reverb_fb_low_shelf_db;
    last_fb_hs_hz = reverb_fb_high_shelf_hz; last_fb_hs_db = reverb_fb_high_shelf_db;
    fb_hs_hz_morph = reverb_fb_high_shelf_hz;
    pre_hp_coeff   = 1.0f - expf(-TWO_PI_F * PRE_HP_FREQ_HZ / (float)REVERB_FS_HZ);
    duck_atk_coeff = 1.0f - expf(-1.0f / (DUCK_ATK_SEC * (float)REVERB_FS_HZ));
    t0_tap_mod_phase = 0.0f;

    /* Morph state — initialise to targets so the first block produces a
     * meaningful output (avoids "fading up from zero" silence on boot).
     * Density is fixed at MAX, so there is no count morph. */
    t0_window_morph = reverb_t0_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t1_window_morph = reverb_t1_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t2_window_morph = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t0_tap_comp_morph = 1.0f;
    t1_tap_comp_morph = 1.0f;
    t2_tap_comp_morph = 1.0f;
    predelay_a_morph = reverb_predelay_a_s * (float)REVERB_FS_HZ;
    predelay_b_morph = reverb_predelay_b_s * (float)REVERB_FS_HZ;

    /* Pre-compute the exponential envelope LUT — exp(-9.21034 × frac) gives
     * an 80 dB drop over the window. Linearly interpolated at lookup. */
    for (int i = 0; i < ENV_FRAC_BINS; i++) {
        float frac = (float)i / (float)(ENV_FRAC_BINS - 1);
        exp_env_lut[i] = expf(-9.21034f * frac);
    }

    velvet_reverb_regenerate_taps();

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
 *   2. for each tap: pre-comp effGain = baseGain × winGain × env
 *      (density is fixed at MAX — there is no per-tap density fade)
 *   3. sum (pre-comp²) → comp = sqrt(full_energy / energy), morphed
 *   4. final effGain = clamp_int16(pre-comp × comp_morph)
 */

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
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T0_FADE_MIN) fade = T0_FADE_MIN;

    float effWin[MAX_T0_TAPS];
    compute_tap_states_mono(t0TapLadder, t0TapLadderCount, count, window, fade,
                            effOffT0, effWin);

    /* T0 has no decay envelope — taper is baked into baseGain. Density fixed
     * at MAX, so no per-tap density fade. */
    float pre[MAX_T0_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float g = (float)t0TapGains[t] * effWin[t];
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
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T1_FADE_MIN) fade = T1_FADE_MIN;

    float effWin[MAX_T1_TAPS];
    compute_tap_states_mono(t1TapLadder, t1TapLadderCount, count, window, fade,
                            effOffT1, effWin);

    /* T1 has no decay envelope — density fixed at MAX. */
    float pre[MAX_T1_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float g = (float)t1TapGains[t] * effWin[t];
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
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T2_FADE_MIN) fade = T2_FADE_MIN;
    float invWindow = (window > 1e-6f) ? 1.0f / window : 0.0f;

    float effWin[MAX_T2_TAPS];
    compute_tap_states_lr(t2TapLadderL, t2TapLadderR, t2TapLadderCount, count,
                          window, fade, effOffT2L, effOffT2R, effWin);

    float pre[MAX_T2_TAPS];
    float energy = 0.0f;
    for (int t = 0; t < count; t++) {
        float frac = (float)effOffT2L[t] * invWindow;
        float g = (float)t2TapGains[t] * effWin[t] * exp_decay_from_frac(frac);
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
#ifdef VELVET_REVERB_HOST
    for (int t = 0; t < count; t++) host_effwin_t2[t] = effWin[t];
#endif
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

    /* Density is fixed at MAX → no count morph. */
    float t0_win_target = reverb_t0_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t1_win_target = reverb_t1_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t2_win_target = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t0_window_morph += ALPHA_WINDOW_MORPH * (t0_win_target - t0_window_morph);
    t1_window_morph += ALPHA_WINDOW_MORPH * (t1_win_target - t1_window_morph);
    t2_window_morph += ALPHA_WINDOW_MORPH * (t2_win_target - t2_window_morph);

    /* Smooth the pre-delay loop lengths so a Decay-macro sweep doesn't
     * pitch-glide / zipper the feedback lines. Feedback/mix/shelf are gain-
     * like and read directly in do_predelay. */
    float pa_target = reverb_predelay_a_s * (float)REVERB_FS_HZ;
    float pb_target = reverb_predelay_b_s * (float)REVERB_FS_HZ;
    predelay_a_morph += ALPHA_WINDOW_MORPH * (pa_target - predelay_a_morph);
    predelay_b_morph += ALPHA_WINDOW_MORPH * (pb_target - predelay_b_morph);

    /* Smooth the Tone-macro cutoffs so biquad coeffs track a continuous sweep
     * (no staircase zipper). do_finalize / do_predelay recompute from these. */
    lpf_hz_morph   += ALPHA_WINDOW_MORPH * (reverb_lpf_hz          - lpf_hz_morph);
    hpf_hz_morph   += ALPHA_WINDOW_MORPH * (reverb_hpf_hz          - hpf_hz_morph);
    fb_hs_hz_morph += ALPHA_WINDOW_MORPH * (reverb_fb_high_shelf_hz - fb_hs_hz_morph);
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
#define DB_WINDOW_TARGET   0.50f      /* ms */
#define DB_DURATION_TARGET 0.01f      /* seconds */

static float last_t0_win_target = -1e9f, last_t1_win_target = -1e9f, last_t2_dur_target = -1e9f;
static uint16_t effgains_settle_counter = 0;
static uint16_t effgains_periodic_counter = 0;

static void background_eff_gains_update(void)
{
    /* Watch user-facing targets for movement. Density is fixed at MAX so only
     * the windows are watched. */
    int changed = 0;
    if (fabsf(reverb_t0_window_ms_target - last_t0_win_target) > DB_WINDOW_TARGET) {
        last_t0_win_target = reverb_t0_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t1_window_ms_target - last_t1_win_target) > DB_WINDOW_TARGET) {
        last_t1_win_target = reverb_t1_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t2_duration_s_target - last_t2_dur_target) > DB_DURATION_TARGET) {
        last_t2_dur_target = reverb_t2_duration_s_target; changed = 1;
    }

    if (changed) effgains_settle_counter = EFFGAINS_SETTLE_BLOCKS;

    /* Slow periodic refresh — catches morph-state drift during long
     * stationary periods so effGains don't fossilise out of sync. BUT only
     * when the morph IIRs have NOT yet converged to their targets. Once idle
     * and settled, the morph state == targets and a recompute reproduces the
     * exact same effGains: pure wasted work that adds a ~67 ms-periodic
     * compute spike (up to ~217 µs on the T2 stage) capable of pushing a
     * block past its 667 µs deadline. A missed deadline repeats the previous
     * output block → audible "bitcrush". So skip the refresh when converged. */
    if (++effgains_periodic_counter >= EFFGAINS_PERIODIC_BLOCKS) {
        effgains_periodic_counter = 0;
        if (effgains_settle_counter == 0) {
            const float t0w = reverb_t0_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
            const float t1w = reverb_t1_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
            const float t2w = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
            int converged =
                fabsf(t0w - t0_window_morph) < 1.0f &&
                fabsf(t1w - t1_window_morph) < 1.0f &&
                fabsf(t2w - t2_window_morph) < 1.0f;
            if (!converged) effgains_settle_counter = 1;
        }
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

/* Linear-interpolated read of a float ring at a (possibly fractional, possibly
 * negative) position. Mirrors JS readRingInterp. */
__attribute__((always_inline)) static inline float read_ring_interp(const float *buf, uint32_t mask, float pos)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    float a = buf[(uint32_t)i & mask];
    float b = buf[((uint32_t)i + 1u) & mask];
    return a + f * (b - a);
}

/* triangle/poly sine in [-π, π] — same Taylor form as JS fastSin, used for the
 * pre-delay line modulation (sub-Hz, so the corners are inaudible). */
__attribute__((always_inline)) static inline float fast_sin_pi(float x)
{
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666666f + x2 * 0.00833333f));
}

/* ==== Pre-delay sustain engine ====
 * Two modulated, damped feedback delay lines in front of the cascade. Reads
 * input_ready (int16), works in normalised ±1 float (so duck / node-clamp /
 * shelf math matches the JS prototype exactly), and writes the cascade input
 * into predelay_out (still normalised; clamped to int16 in do_input_write_t0).
 * Mirrors velvet_kernel.js lines 599-663. When feedback is 0 the loop is
 * bypassed and the dry input passes straight through. */
static void do_predelay(void)
{
    const float inv_fs = 1.0f / (float)REVERB_FS_HZ;
    const float to_unit = 1.0f / 32768.0f;

    float fb = reverb_feedback;
    if (fb > 0.999f) fb = 0.999f;
    if (fb < 0.0f) fb = 0.0f;

    float timeA = predelay_a_morph;
    float timeB = predelay_b_morph;
    if (timeA < PRE_MIN_TIME_SAMPLES) timeA = PRE_MIN_TIME_SAMPLES;
    if (timeA > (float)PRE_DELAY_MAX_SAMPLES) timeA = (float)PRE_DELAY_MAX_SAMPLES;
    if (timeB < PRE_MIN_TIME_SAMPLES) timeB = PRE_MIN_TIME_SAMPLES;
    if (timeB > (float)PRE_DELAY_MAX_SAMPLES) timeB = (float)PRE_DELAY_MAX_SAMPLES;

    float modDepth = reverb_fb_mod_depth;
    if (modDepth < 0.0f) modDepth = 0.0f; if (modDepth > 1.0f) modDepth = 1.0f;
    modDepth *= (float)PRE_MOD_MAX_SAMPLES;
    float modRate = reverb_fb_mod_rate; if (modRate < 0.0f) modRate = 0.0f;
    float modInc = TWO_PI_F * modRate * inv_fs;

    float mix = reverb_delay_mix;
    if (mix < 0.0f) mix = 0.0f; else if (mix > 1.0f) mix = 1.0f;

    float duckAmount = reverb_duck_amount;
    if (duckAmount < 0.0f) duckAmount = 0.0f; else if (duckAmount > 1.0f) duckAmount = 1.0f;
    float duckRelSec = reverb_duck_release_s; if (duckRelSec < 0.01f) duckRelSec = 0.01f;
    float duckRelCoef = 1.0f - expf(-1.0f / (duckRelSec * (float)REVERB_FS_HZ));

    /* Shared shelf coeffs — recompute (for both lines) only on change. */
    if (reverb_fb_low_shelf_hz != last_fb_ls_hz || reverb_fb_low_shelf_db != last_fb_ls_db) {
        biquad_lowshelf(&pre_low_shelf_a, reverb_fb_low_shelf_hz, reverb_fb_low_shelf_db, (float)REVERB_FS_HZ);
        biquad_lowshelf(&pre_low_shelf_b, reverb_fb_low_shelf_hz, reverb_fb_low_shelf_db, (float)REVERB_FS_HZ);
        last_fb_ls_hz = reverb_fb_low_shelf_hz; last_fb_ls_db = reverb_fb_low_shelf_db;
    }
    /* Tone-mapped high shelf sits IN the feedback loop, so a raw coeff jump
     * gets sustained/amplified — smooth the cutoff and use a small threshold
     * (db is fixed, compared directly). */
    if (fabsf(fb_hs_hz_morph - last_fb_hs_hz) > last_fb_hs_hz * 0.0003f ||
        reverb_fb_high_shelf_db != last_fb_hs_db) {
        biquad_highshelf(&pre_high_shelf_a, fb_hs_hz_morph, reverb_fb_high_shelf_db, (float)REVERB_FS_HZ);
        biquad_highshelf(&pre_high_shelf_b, fb_hs_hz_morph, reverb_fb_high_shelf_db, (float)REVERB_FS_HZ);
        last_fb_hs_hz = fb_hs_hz_morph; last_fb_hs_db = reverb_fb_high_shelf_db;
    }

    if (fb <= 0.0f) {
        /* Bypass: dry input straight to the cascade (loop off). */
        for (int n = 0; n < REVERB_BLOCK; n++)
            predelay_out[n] = (float)input_ready[n] * to_unit;
        /* Keep the write head advancing so resuming feedback reads sane history. */
        predelay_wh += REVERB_BLOCK;
        if (predelay_wh >= REVERB_IDX_WRAP) predelay_wh -= REVERB_IDX_WRAP;
        return;
    }

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float px = (float)input_ready[n] * to_unit;
        float rect = px < 0.0f ? -px : px;
        if (rect > duck_env) duck_env += duck_atk_coeff * (rect - duck_env);
        else                 duck_env += duckRelCoef   * (rect - duck_env);
        float duck = duckAmount * duck_env; if (duck > 1.0f) duck = 1.0f;
        float fbEff = fb * (1.0f - duck);

        /* line A */
        pre_mod_phase_a += modInc; if (pre_mod_phase_a >= TWO_PI_F) pre_mod_phase_a -= TWO_PI_F;
        float pa = pre_mod_phase_a; if (pa > PI_F) pa -= TWO_PI_F;
        float extraA = modDepth * (1.0f + fast_sin_pi(pa));
        float rA = read_ring_interp(predelay_a, PRE_DELAY_LINE_MASK,
                                    (float)predelay_wh - timeA - extraA);
        rA = biquad_process(&pre_high_shelf_a, biquad_process(&pre_low_shelf_a, rA));
        pre_hp_a += pre_hp_coeff * (rA - pre_hp_a);
        float nodeA = px + fbEff * (rA - pre_hp_a);
        if (nodeA > 4.0f) nodeA = 4.0f; else if (nodeA < -4.0f) nodeA = -4.0f;
        predelay_a[predelay_wh & PRE_DELAY_LINE_MASK] = nodeA;

        /* line B (incommensurate time + decorrelated mod phase) */
        pre_mod_phase_b += modInc; if (pre_mod_phase_b >= TWO_PI_F) pre_mod_phase_b -= TWO_PI_F;
        float pb = pre_mod_phase_b; if (pb > PI_F) pb -= TWO_PI_F;
        float extraB = modDepth * (1.0f + fast_sin_pi(pb));
        float rB = read_ring_interp(predelay_b, PRE_DELAY_LINE_MASK,
                                    (float)predelay_wh - timeB - extraB);
        rB = biquad_process(&pre_high_shelf_b, biquad_process(&pre_low_shelf_b, rB));
        pre_hp_b += pre_hp_coeff * (rB - pre_hp_b);
        float nodeB = px + fbEff * (rB - pre_hp_b);
        if (nodeB > 4.0f) nodeB = 4.0f; else if (nodeB < -4.0f) nodeB = -4.0f;
        predelay_b[predelay_wh & PRE_DELAY_LINE_MASK] = nodeB;

        predelay_out[n] = px + mix * (nodeA + nodeB - px);
        predelay_wh++;
        if (predelay_wh >= REVERB_IDX_WRAP) predelay_wh -= REVERB_IDX_WRAP;
    }
}

static void do_input_write_t0(void)
{
    /* Attenuate the (pre-delayed) input into the cascade by REVERB_HEADROOM so
     * the whole int16 chain runs cooler; the output stage makes it back up.
     * predelay_out is normalised ±1 but blooms well past it during sustain
     * (the pre-delay node clamp is ±4, so predelay_out can reach ~±5). Use the
     * SAME soft knee as every other stage bridge (soft_clip_int16) rather than
     * a hard clamp: the JS reference feeds a float ring here and never clips, so
     * a hard int16 clamp injects an edge that the cascade diffuses into an
     * audible "whoosh". The soft knee rounds those occasional overs off
     * smoothly, matching the JS behaviour as closely as int16 allows. */
    for (int i = 0; i < REVERB_BLOCK; i++) {
        uint32_t wi = ring_write_idx + (uint32_t)i;
        t0_ring[wi & T0_RING_MASK] = soft_clip_int16(predelay_out[i] * (32768.0f * REVERB_HEADROOM));
    }
    block_write_idx = ring_write_idx;
    ring_write_idx += REVERB_BLOCK;
    /* Keep the free-running index in float-exact range. block_write_idx is cast to
     * float in the recirc/global read-position math (pos0 = (float)block_write_idx
     * - modOff); past 2^24 a float32 can't represent consecutive sample indices, so
     * the feedback tap positions quantize — audible as a slow "bit-crush" / sample-
     * rate-reduction after ~12 min of running. Wrap at 2^23, a multiple of every
     * (power-of-2) ring size, so the masked ring positions stay perfectly continuous
     * across the wrap and floats stay exact indefinitely. */
    if (ring_write_idx >= REVERB_IDX_WRAP) ring_write_idx -= REVERB_IDX_WRAP;

    for (int i = 0; i < REVERB_BLOCK; i++) accT0[i] = 0;
}

/* Fast (SIMD pair-read) T0 convolution — used when the per-tap LFO is off. */
/* Wrap-aware Q15 SIMD accumulation of one delay-line tap into accT0:
 * accT0[n] += t0_ring[(base + n) & mask] * gain  for n in [0, REVERB_BLOCK).
 * `base` must already be masked into [0, T0_RING_SAMPLES). Shared by the fast
 * (unmodulated) path and the modulated path's two interpolation passes. */
static void accumulate_t0(uint32_t base, uint32_t gain)
{
    /* Common case: the whole block reads contiguously without wrapping the
     * ring. Keep this a tight, fixed-count SIMD loop (REVERB_BLOCK is a small
     * even constant) so the compiler fully unrolls it — this is the hot path. */
    if (T0_RING_SAMPLES - base >= (uint32_t)REVERB_BLOCK) {
        const int16_t *src = &t0_ring[base];
        for (int i = 0; i < REVERB_BLOCK; i += 2) {
            uint32_t s_pair = *((const u32_alias *)(src + i));
            accT0[i]   = qadd_sat(accT0[i],   smulbb(s_pair, gain));
            accT0[i+1] = qadd_sat(accT0[i+1], smultb(s_pair, gain));
        }
        return;
    }
    /* Rare case: the read window straddles the ring wrap. Segment 1 is the
     * contiguous run of n1 samples from base; segment 2 reads from ring start.
     * Pairs are only read within a segment, so a pair never straddles the wrap
     * (which would over-read / over-write accT0); the odd boundary sample of
     * each segment is handled with a scalar multiply. */
    uint32_t n1 = T0_RING_SAMPLES - base;
    int i = 0;
    for (; i + 1 < (int)n1; i += 2) {
        uint32_t s_pair = *((const u32_alias *)(&t0_ring[base + (uint32_t)i]));
        accT0[i]   = qadd_sat(accT0[i],   smulbb(s_pair, gain));
        accT0[i+1] = qadd_sat(accT0[i+1], smultb(s_pair, gain));
    }
    if (i < (int)n1) {
        accT0[i] = qadd_sat(accT0[i], smulbb((uint32_t)(uint16_t)t0_ring[base + (uint32_t)i], gain));
        i++;
    }
    for (; i + 1 < REVERB_BLOCK; i += 2) {
        uint32_t s_pair = *((const u32_alias *)(&t0_ring[(uint32_t)i - n1]));
        accT0[i]   = qadd_sat(accT0[i],   smulbb(s_pair, gain));
        accT0[i+1] = qadd_sat(accT0[i+1], smultb(s_pair, gain));
    }
    if (i < REVERB_BLOCK) {
        accT0[i] = qadd_sat(accT0[i], smulbb((uint32_t)(uint16_t)t0_ring[(uint32_t)i - n1], gain));
    }
}

static inline void do_t0_phase_fast(void)
{
    int tapEnd = t0TapCount;
    for (int t = 0; t < tapEnd; t++) {
        uint32_t gain = (uint32_t)(int32_t)effGains_t0[t];
        uint32_t base = (block_write_idx - (uint32_t)effOffT0[t]) & T0_RING_MASK;
        accumulate_t0(base, gain);
    }
}

/* Fractionally-delayed, gain-scaled accumulation of one tap into accT0 in a
 * single pass. The (REVERB_BLOCK+1)-sample window starting at `base` is read
 * once; for each output sample a packed pair {ring[base+k], ring[base+k+1]} is
 * combined with the packed gains `gab` = {ga, gb} via one SMLAD:
 *   accT0[k] += ring[base+k]·ga + ring[base+k+1]·gb
 * which is the 2-tap linear interpolation (ga = g·rf, gb = g·(1-rf)) folded
 * into the tap gain. Replaces the old two-pass scheme (2× ring reads + 2×
 * accT0 read-modify-write per tap). The internal SMLAD sum can't overflow:
 * |ring·ga + ring·gb| ≤ 32767·(ga+gb) = 32767·g ≤ 32767². */
static void accumulate_t0_interp(uint32_t base, uint32_t gab)
{
    /* Point `src` at a contiguous (REVERB_BLOCK+1)-sample window. In the common
     * case that window lies inside the ring without wrapping, so we read it
     * directly (no copy). Only when it straddles the wrap do we gather it once
     * into a small local buffer; either way the SMLAD loop below is identical. */
    const int16_t *src;
    int16_t w[REVERB_BLOCK + 1];
    if (T0_RING_SAMPLES - base >= (uint32_t)REVERB_BLOCK + 1u) {
        src = &t0_ring[base];
    } else {
        for (int k = 0; k <= REVERB_BLOCK; k++)
            w[k] = t0_ring[(base + (uint32_t)k) & T0_RING_MASK];
        src = w;
    }
    for (int k = 0; k < REVERB_BLOCK; k++) {
        uint32_t wpair = *((const u32_alias *)(src + k));
        accT0[k] = qadd_sat(accT0[k], smlad(wpair, gab, 0));
    }
}

/* Modulated T0 convolution — each tap's read position is displaced by a
 * per-tap (golden-angle phase) LFO so the early-reflection cluster shimmers
 * (Lexicon-style chorused early reflections). Mirrors JS accumOffMod: extra ∈
 * [0, 2·depth] keeps reads in the past.
 *
 * The modulation offset is held constant across the 16-sample block: a slow
 * chorus LFO moves < 1 sample over 0.67 ms, so a zero-order hold at block rate
 * (1500 Hz) is inaudible while letting us drop the per-sample float interp +
 * serial sin/cos recurrence (the old hot loop, ~657 µs) for the cheap Q15 SIMD
 * path. Each tap's fractional delay (idel + rf) is realised as a 2-tap linear
 * interpolation folded into the tap gain and applied in a single SMLAD pass
 * (see accumulate_t0_interp):
 *   accT0[n] += g·rf · ring[ri_base-1+n] + g·(1-rf) · ring[ri_base+n] */
static void do_t0_phase_mod(float depth)
{
    int tapEnd = t0TapCount;
    float ph0 = t0_tap_mod_phase;
    float sinP0 = sinf(ph0), cosP0 = cosf(ph0);
    for (int t = 0; t < tapEnd; t++) {
        int32_t g = (int32_t)effGains_t0[t];
        if (g == 0) continue;
        /* per-tap LFO sin (rotate base phase by the tap's golden-angle offset) */
        float s = sinP0 * t0TapModCos[t] + cosP0 * t0TapModSin[t];
        float delta = depth * (1.0f + s);          /* extra delay, always ≥ 0 */
        int32_t idel = (int32_t)delta;              /* floor (delta ≥ 0) */
        float   rf   = delta - (float)idel;         /* fractional part ∈ [0,1) */
        int32_t ga   = (int32_t)((float)g * rf);    /* gain for older sample (ri_base-1) */
        int32_t gb   = g - ga;                       /* gain for ri_base (conserves g) */
        /* pack {ga (low half), gb (high half)} to pair with {ring[k], ring[k+1]} */
        uint32_t gab = ((uint32_t)(uint16_t)gb << 16) | ((uint32_t)(uint16_t)ga & 0xFFFFu);
        uint32_t base = (block_write_idx - (uint32_t)effOffT0[t] - (uint32_t)idel - 1u) & T0_RING_MASK;
        accumulate_t0_interp(base, gab);
    }
}

static void do_t0_phase(void)
{
    float depth = reverb_t0_tap_mod_depth;
    if (depth < 0.0f) depth = 0.0f; if (depth > 1.0f) depth = 1.0f;
    depth *= T0_TAP_MOD_MAX_SAMPLES;
    if (depth <= 0.0f) {
        do_t0_phase_fast();
        return;
    }
    float rate = reverb_t0_tap_mod_rate; if (rate < 0.0f) rate = 0.0f;
    float inc = TWO_PI_F * rate / (float)REVERB_FS_HZ;
    do_t0_phase_mod(depth);
    /* Advance the shared LFO phase by the block. */
    t0_tap_mod_phase += (float)REVERB_BLOCK * inc;
    t0_tap_mod_phase = fmodf(t0_tap_mod_phase, TWO_PI_F);
    if (t0_tap_mod_phase < 0.0f) t0_tap_mod_phase += TWO_PI_F;
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

static void do_t2_phase(void)
{
    int count = t2TapCount;
#ifdef DIAG_REVERB_PROFILE
    t2_dma_wait_cyc = 0;
    t2_dma_kick_cyc = 0;
#endif
    if (count == 0) return;

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

#ifdef DIAG_REVERB_PROFILE
    DIAG_LOG_BLK(DIAG_EVT_T2_DMA_WAIT, t2_dma_wait_cyc);
    DIAG_LOG_BLK(DIAG_EVT_T2_DMA_KICK, t2_dma_kick_cyc);
#endif
}

/* Pre-T2 tanh saturation. Applied to the freshly-written block of t2_ring
 * (Tm bridge output) so all future T2 tap reads see the saturated signal.
 * Mirrors the JS prototype's "Pre T2"
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

static void do_finalize(void)
{
    /* Biquad coefficient recompute is expensive (~500 cycles per call —
     * sinf + cosf + division). With macros tied to ADC pots, raw `!=` would
     * recompute every block from sample-rate ADC noise. Hysteresis at 0.5 %
     * of the current freq is well below audibility (≪ ¼ semitone) but
     * caps recompute rate to a handful per second during a smooth sweep. */
    /* Track the SMOOTHED cutoff (lpf_hz_morph), recomputing whenever it has
     * moved a hair. The morph slews continuously (~30 ms tau) so this fires
     * every block during a sweep — each step is tiny and follows a smooth
     * curve (no zipper) — then stops once converged (no idle cost). The 0.03 %
     * threshold is ≪ audibility but coarse enough to halt at convergence. */
    float lpf_target = lpf_hz_morph;
    if (lpf_target < 200.0f) lpf_target = 200.0f;
    if (lpf_target > 11900.0f) lpf_target = 11900.0f;
    if (fabsf(lpf_target - last_lpf_hz) > last_lpf_hz * 0.0003f) {
        last_lpf_hz = lpf_target;
        biquad_lpf(&lpf_L, lpf_target, 48000.0f);
        biquad_lpf(&lpf_R, lpf_target, 48000.0f);
    }
    float hpf_target = hpf_hz_morph;
    if (hpf_target < 20.0f)   hpf_target = 20.0f;
    if (hpf_target > 2000.0f) hpf_target = 2000.0f;
    if (fabsf(hpf_target - last_hpf_hz) > last_hpf_hz * 0.0003f) {
        last_hpf_hz = hpf_target;
        biquad_hpf(&hpf_L, hpf_target, 48000.0f);
        biquad_hpf(&hpf_R, hpf_target, 48000.0f);
    }

    uint8_t write_buf = (uint8_t)(blocks_produced % REVERB_OUT_NBUF);
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

    /* Publish: the block is fully written before blocks_produced exposes it. */
    __DMB();
    blocks_produced++;
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
#ifndef VELVET_REVERB_HOST
            block_ready_cyc = DWT->CYCCNT;
#endif
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

    /* DIAGNOSTIC RESULT (reverb send = 0): clicks DISAPPEAR. The clicks are
     * in the reverb path — either the reverb amplifying a step in its input
     * (it is fed the delay `mix`, so any gain/mode step zippers the input and
     * the convolution rings it into an audible click that is inaudible in the
     * dry path) or an output-ring underrun. Supersedes the earlier
     * poll-short-circuit test, which was confounded. */

#ifndef VELVET_REVERB_HOST
#ifdef DIAG_REVERB_PROFILE
    /* Pick the one-in-N blocks that get fully instrumented this pass. Chosen
     * before any timing starts so every stage below agrees on whether it is
     * being sampled, which is what lets the per-stage numbers be summed and
     * reconciled against the block total. */
    /* Modulo, not a power-of-two mask, so the divisor can be chosen coprime with
     * the other periodic work in the block. With a mask the divisor must be a
     * power of two, and 128 shares all its factors with the 16-block macro
     * throttle in update_morph_state: the sampled blocks then land on the same
     * phase every time and the sample is not of the block population but of one
     * particular kind of block. That aliasing is what made reverb_morph look far
     * more expensive than it is. A prime divisor cannot phase-lock to anything. */
    static uint32_t prof_block_ctr = 0;
    diag_sample_block = ((++prof_block_ctr % DIAG_PROFILE_DIVISOR) == 0u);
#endif
    uint32_t i_blk_0; uint32_t t0 = diag_stage_start(&i_blk_0);
    /* Wall-clock lag from block-ready to poll pickup: how long the fresh input
     * block sat waiting on the main loop (includes any ISR + other main-loop
     * work that delayed the reverb). Large values => main loop not reaching
     * poll promptly, which lets the output reader run dry. */
    diag_log(DIAG_EVT_POLL_LATENCY, t0 - block_ready_cyc);
    /* Emit the cumulative bitcrush miss count ~once/sec (1500 blocks/s). */
    static uint16_t miss_emit_ctr = 0;
    if (++miss_emit_ctr >= 1500u) { miss_emit_ctr = 0; diag_log(DIAG_EVT_OUTPUT_MISS, output_miss_count); }
#endif

    /* Cheap per-block morph update (window/decay IIR + pre-delay time smooth).
     * The heavier effGains recompute is deferred to outside the block timer. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_morph_0; uint32_t t_morph_0 = diag_stage_start(&i_morph_0);
#endif
    update_morph_state();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_MORPH, diag_stage_cycles(t_morph_0, i_morph_0));
#endif

#if REVERB_STAGE >= 1
    /* Pre-delay sustain engine runs on the input block, producing the
     * (feedforward) cascade input. Replaces the old per-stage/global recirc. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_pre_0; uint32_t t_pre_0 = diag_stage_start(&i_pre_0);
#endif
    do_predelay();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_PREDELAY, diag_stage_cycles(t_pre_0, i_pre_0));
#endif
    do_input_write_t0();
#endif
#if REVERB_STAGE >= 2
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i0_t0; uint32_t t0_t0 = diag_stage_start(&i0_t0);
#endif
        do_t0_phase();
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T0, diag_stage_cycles(t0_t0, i0_t0));
#endif
        do_bridge_t0_to_t1();
    }
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i1_t0; uint32_t t1_t0 = diag_stage_start(&i1_t0);
#endif
        do_t1_phase();
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T1, diag_stage_cycles(t1_t0, i1_t0));
#endif
        do_bridge_t1_to_t2();
    }
#endif
#if REVERB_STAGE >= 3
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i2_t0; uint32_t t2_t0 = diag_stage_start(&i2_t0);
#endif
        do_t2_phase();
        /* Pre-T2 tanh sat — saturates the Tm bridge content in t2_ring before
         * subsequent block reads it. Runs unconditionally (always-on). */
#ifndef DIAG_BISECT_NO_T2SAT
        /* Bisection: this rewrites ring content in place that 32 T2 taps go on
         * re-reading for the next 10.9 s, so a fault here would sound reverberant
         * rather than like a bare click. */
        apply_pre_t2_sat();
#endif
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T2, diag_stage_cycles(t2_t0, i2_t0));
#endif
    }
#endif
#if REVERB_STAGE >= 4
#ifndef VELVET_REVERB_HOST
    uint32_t i_fin_0; uint32_t t_fin_0 = diag_stage_start(&i_fin_0);
#endif
    do_finalize();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_FINALIZE, diag_stage_cycles(t_fin_0, i_fin_0));
#endif
#endif

#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_BLOCK, DWT->CYCCNT - t0);
    /* ISR cycles that preempted this block (wall-clock block − this = pure reverb compute). */
    DIAG_LOG_BLK(DIAG_EVT_REVERB_ISR_IN_BLOCK, diag_isr_cycles - i_blk_0);
#endif

    block_ready = 0;

    /* Background update — NOT inside the block timer but timed separately
     * so we can see how much CPU it eats between blocks. Whatever it costs
     * is unavailable to the next block; if it stretches past the input
     * buffer's lead time we miss a deadline even though the per-stage
     * numbers look fine. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_bg_0; uint32_t t_bg_0 = diag_stage_start(&i_bg_0);
#endif
    background_eff_gains_update();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_BG_EFF, diag_stage_cycles(t_bg_0, i_bg_0));
#endif
}

/* ==== Output ====
 * Each channel pulls one block per wrap from the output ring, holding a steady
 * ~1-block lag behind the newest published block so a single late poll doesn't
 * starve it. The two codec ISRs track the writer independently — no shared
 * read state, no order dependence between them.
 *
 * Lag policy (let newest = blocks_produced - 1 = newest published block, and
 * lag = newest - rd_seq = completed blocks buffered ahead of the reader):
 *   - advance the read sequence by one block per wrap;
 *   - if it passes newest (underrun: cushion fully drained), clamp to newest
 *     and count an output_miss (replays one block);
 *   - if lag grows beyond NBUF-1 (writer about to lap the reader), resync to
 *     the target lag, dropping stale audio.
 * Crucially the reader does NOT hug newest — it free-runs at one block per
 * wrap so the cushion floats around TARGET_LAG, absorbing poll jitter. The
 * reader thus reads slot (rd_seq % NBUF): always a completed block, never the
 * writer's in-progress slot. */
int16_t velvet_reverb_out_left(void)
{
    if (out_idx_L == 0) {
        uint32_t wr = blocks_produced;
        if (wr != 0u) {
            uint32_t newest = wr - 1u;
            rd_seq_L++;
            int32_t lag = (int32_t)(newest - rd_seq_L);
            if (lag < 0) {
                rd_seq_L = newest;
#ifndef VELVET_REVERB_HOST
                output_miss_count++;
#endif
            } else if (lag > (REVERB_OUT_NBUF - 1)) {
                rd_seq_L = newest - REVERB_OUT_TARGET_LAG;
            }
            reader_buf_L = (uint8_t)(rd_seq_L % REVERB_OUT_NBUF);
        }
    }
    int16_t s = outL[reader_buf_L][out_idx_L];
    out_idx_L = (uint8_t)((out_idx_L + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}

int16_t velvet_reverb_out_right(void)
{
    if (out_idx_R == 0) {
        uint32_t wr = blocks_produced;
        if (wr != 0u) {
            uint32_t newest = wr - 1u;
            rd_seq_R++;
            int32_t lag = (int32_t)(newest - rd_seq_R);
            if (lag < 0) {
                rd_seq_R = newest;
            } else if (lag > (REVERB_OUT_NBUF - 1)) {
                rd_seq_R = newest - REVERB_OUT_TARGET_LAG;
            }
            reader_buf_R = (uint8_t)(rd_seq_R % REVERB_OUT_NBUF);
        }
    }
    int16_t s = outR[reader_buf_R][out_idx_R];
    out_idx_R = (uint8_t)((out_idx_R + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}

#ifdef VELVET_REVERB_HOST
/* ==== Host-only test hooks: tap relocation / uniformity probe ====
 * Set a stage's morphed window directly and recompute its effGains/effOff so
 * the host harness can inspect the tap relocation ladder (analogue of
 * test/velvet_relocate.js). Settles the gain-comp morph by recomputing a few
 * times so effGain magnitudes are stable. Density is fixed at MAX. */
int  host_t2_tap_count(void)         { return t2TapCount; }
uint16_t host_t2_effoff_l(int i)     { return effOffT2L[i]; }
int16_t  host_t2_effgain(int i)      { return effGains_t2[i]; }
float    host_t2_effwin(int i)       { return host_effwin_t2[i]; }
void host_t2_set_window_samples(float w)
{
    t2_window_morph = w;
    for (int s = 0; s < 32; s++) recompute_eff_gains_t2();
}
float host_t2_duration_max_samples(void) { return (float)T2_DURATION_MAX_SAMPLES; }
float host_t2_min_window_samples(void)   { return 0.2f * (float)REVERB_FS_HZ; }
#endif
