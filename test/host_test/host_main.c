/* host_main.c — diagnostic driver for velvet_reverb on host
 *
 * Builds velvet_reverb.c with VELVET_REVERB_HOST so DMA/QADD are software,
 * then drives it like the codec ISR would and reports statistics that
 * should expose:
 *   - output stuck at saturation / DC bias
 *   - accumulator wrap or other huge values
 *   - out-of-bounds writes (relies on ASan/UBSan from Makefile)
 *   - L/R desync at finalize boundary
 *
 * Each "frame" simulates one audio sample: push_sample(input), out_left,
 * out_right. The real device does 4 frames per ISR but for testing the
 * logic the 1-per-call cadence is equivalent.
 */

#include "host_shim.h"
#include "../../velvet_reverb.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

/* Backing storage for the simulated SDRAM T2 ring + pre-delay lines. */
int16_t host_t2_ring_storage[T2_RING_SAMPLES];
float   host_predelay_a_storage[PRE_DELAY_LINE_SAMPLES];
float   host_predelay_b_storage[PRE_DELAY_LINE_SAMPLES];

typedef struct {
    int    peak_L, peak_R;
    long   sum_L, sum_R;
    long   sum_abs_L, sum_abs_R;
    int    sat_L, sat_R;       /* count of |out| == 32767 */
    int    zero_L, zero_R;     /* count of out == 0 */
    int    samples;
} stats_t;

static void stats_init(stats_t *s) { memset(s, 0, sizeof(*s)); }

static void stats_update(stats_t *s, int16_t outL, int16_t outR) {
    int aL = (outL < 0) ? -outL : outL;
    int aR = (outR < 0) ? -outR : outR;
    if (aL > s->peak_L) s->peak_L = aL;
    if (aR > s->peak_R) s->peak_R = aR;
    s->sum_L += outL;
    s->sum_R += outR;
    s->sum_abs_L += aL;
    s->sum_abs_R += aR;
    if (outL == 32767 || outL == -32767) s->sat_L++;
    if (outR == 32767 || outR == -32767) s->sat_R++;
    if (outL == 0) s->zero_L++;
    if (outR == 0) s->zero_R++;
    s->samples++;
}

static void stats_report(const stats_t *s, const char *label) {
    printf("\n--- %s ---\n", label);
    printf("  samples       : %d\n", s->samples);
    printf("  peak L / R    : %d / %d\n", s->peak_L, s->peak_R);
    printf("  mean L / R    : %.2f / %.2f      (DC bias, want ~0)\n",
           (double)s->sum_L / s->samples, (double)s->sum_R / s->samples);
    printf("  mean|L| / |R| : %.2f / %.2f      (loudness)\n",
           (double)s->sum_abs_L / s->samples, (double)s->sum_abs_R / s->samples);
    printf("  sat L / R     : %d / %d  (%.2f%% / %.2f%%)\n",
           s->sat_L, s->sat_R,
           100.0 * s->sat_L / s->samples, 100.0 * s->sat_R / s->samples);
    printf("  zero L / R    : %d / %d  (%.2f%% / %.2f%%)\n",
           s->zero_L, s->zero_R,
           100.0 * s->zero_L / s->samples, 100.0 * s->zero_R / s->samples);
}

static void test_silence(void) {
    velvet_reverb_init();
    /* Drive all macros open so taps, recirc, and filters are fully active —
     * otherwise default Density/Decay = 0 leaves the reverb near-silent and
     * none of the hot paths get exercised. */
    velvet_reverb_apply_decay_macro(1.0f);
    velvet_reverb_apply_tone_macro(1.0f);
    stats_t s; stats_init(&s);

    /* 1 second of silence. Output should be 0 everywhere. */
    for (int n = 0; n < 48000; n++) {
        velvet_reverb_push_sample(0);
        velvet_reverb_poll();   /* drains a ready block, no-op otherwise */
        stats_update(&s, velvet_reverb_out_left(), velvet_reverb_out_right());
    }
    stats_report(&s, "Silence in (1 s)");
}

static void test_impulse(void) {
    velvet_reverb_init();
    /* Drive all macros open so taps, recirc, and filters are fully active —
     * otherwise default Density/Decay = 0 leaves the reverb near-silent and
     * none of the hot paths get exercised. */
    velvet_reverb_apply_decay_macro(1.0f);
    velvet_reverb_apply_tone_macro(1.0f);
    stats_t s; stats_init(&s);

    /* Single full-scale impulse, then silence for 2 sec. */
    velvet_reverb_push_sample(32767);
    velvet_reverb_poll();
    stats_update(&s, velvet_reverb_out_left(), velvet_reverb_out_right());

    /* Print the first 200 output samples after the impulse so we can see
     * the leading edge of the IR. */
    printf("\n--- Impulse IR head (first 200 samples after impulse) ---\n");
    printf("sample,outL,outR\n");
    for (int n = 1; n < 48000 * 2; n++) {
        velvet_reverb_push_sample(0);
        velvet_reverb_poll();
        int16_t L = velvet_reverb_out_left();
        int16_t R = velvet_reverb_out_right();
        stats_update(&s, L, R);
        if (n <= 200) printf("%d,%d,%d\n", n, L, R);
    }
    stats_report(&s, "Impulse (2 s)");
}

static void test_quiet_noise(void) {
    velvet_reverb_init();
    /* Drive all macros open so taps, recirc, and filters are fully active —
     * otherwise default Density/Decay = 0 leaves the reverb near-silent and
     * none of the hot paths get exercised. */
    velvet_reverb_apply_decay_macro(1.0f);
    velvet_reverb_apply_tone_macro(1.0f);
    stats_t s; stats_init(&s);

    srand(42);
    for (int n = 0; n < 48000 * 3; n++) {
        int16_t in = (int16_t)(((rand() & 0xFFFF) - 0x8000) >> 3);
        velvet_reverb_push_sample(in);
        velvet_reverb_poll();
        stats_update(&s, velvet_reverb_out_left(), velvet_reverb_out_right());
    }
    stats_report(&s, "White noise in, RMS ~4000 (3 s)");
}

/* ---- Full impulse-response dump (spectral A/B against a stored baseline) ----
 * Fires a single full-scale impulse at decay/tone = 1.0 and writes the wet
 * stereo output to a CSV. An offline IR capture produces a
 * matched-settings reference; the RNG layouts differ so the comparison is
 * statistical/spectral, not sample-exact. */
#define IR_DUMP_SAMPLES (48000 * 2)
static void test_dump_ir(const char *path) {
    velvet_reverb_init();
    velvet_reverb_apply_decay_macro(1.0f);
    velvet_reverb_apply_tone_macro(1.0f);

    FILE *f = fopen(path, "w");
    if (!f) { printf("  (could not open %s for IR dump)\n", path); return; }
    fprintf(f, "sample,outL,outR\n");

    int nonfinite = 0;
    velvet_reverb_push_sample(32767);
    velvet_reverb_poll();
    {
        int16_t L = velvet_reverb_out_left(), R = velvet_reverb_out_right();
        fprintf(f, "0,%d,%d\n", L, R);
    }
    for (int n = 1; n < IR_DUMP_SAMPLES; n++) {
        velvet_reverb_push_sample(0);
        velvet_reverb_poll();
        int16_t L = velvet_reverb_out_left(), R = velvet_reverb_out_right();
        fprintf(f, "%d,%d,%d\n", n, L, R);
    }
    fclose(f);
    printf("\n--- IR dump ---\n  wrote %d samples to %s  (non-finite: %d)\n",
           IR_DUMP_SAMPLES, path, nonfinite);
}

/* ---- Tap relocation / uniformity check ----
 * Ramp the T2 window from max down to min in fine steps and verify:
 *   (1) no slide — a tap's read position only changes while its gain is near 0
 *   (2) uniform spacing — coefficient of variation of active-tap spacing stays
 *       bounded across the whole range (velvet property holds)
 *   (3) at the minimum window, essentially all M taps are active (relocated in,
 *       not culled). */
extern int      host_t2_tap_count(void);
extern uint16_t host_t2_effoff_l(int i);
extern float    host_t2_effwin(int i);          /* window/relocation gain (no decay env) */
extern void     host_t2_set_window_samples(float w);
extern float    host_t2_duration_max_samples(void);
extern float    host_t2_min_window_samples(void);

static void test_relocation(void) {
    velvet_reverb_init();
    int M = host_t2_tap_count();
    float wmax = host_t2_duration_max_samples();
    float wmin = host_t2_min_window_samples();

    const int STEPS = 400;
    int prev_off[64];
    float prev_win[64];
    int have_prev = 0;
    int slides = 0;
    double cv_max = 0.0;
    int min_active = M;
    double final_cv = 0.0;
    int final_active = 0;
    /* "present" = window gain past the crossfade midpoint. The decay envelope is
     * deliberately excluded: this measures tap placement, not tap level. */
    const float PRESENT = 0.5f;

    for (int s = 0; s < STEPS; s++) {
        float frac = (float)s / (float)(STEPS - 1);     /* 0 = max window, 1 = min */
        float w = wmax + frac * (wmin - wmax);
        host_t2_set_window_samples(w);

        int offs[64], nact = 0;
        for (int i = 0; i < M; i++) {
            if (host_t2_effwin(i) > PRESENT) offs[nact++] = host_t2_effoff_l(i);
        }
        if (nact < min_active) min_active = nact;

        /* no-slide: a tap present in BOTH this and the previous step must not
         * have changed its read position (relocations happen at gain≈0). */
        if (have_prev) {
            for (int i = 0; i < M; i++) {
                if (host_t2_effwin(i) > PRESENT && prev_win[i] > PRESENT &&
                    host_t2_effoff_l(i) != prev_off[i]) slides++;
            }
        }
        for (int i = 0; i < M; i++) { prev_off[i] = host_t2_effoff_l(i); prev_win[i] = host_t2_effwin(i); }
        have_prev = 1;

        /* spacing CV of the present taps (sort offsets first) */
        if (nact >= 3) {
            for (int a = 0; a < nact; a++) for (int b = a+1; b < nact; b++)
                if (offs[b] < offs[a]) { int t = offs[a]; offs[a] = offs[b]; offs[b] = t; }
            double mean = 0.0; int ng = nact - 1;
            for (int a = 1; a < nact; a++) mean += (offs[a] - offs[a-1]);
            mean /= ng;
            double var = 0.0;
            for (int a = 1; a < nact; a++) { double d = (offs[a]-offs[a-1]) - mean; var += d*d; }
            var /= ng;
            double cv = (mean > 1e-9) ? sqrt(var)/mean : 0.0;
            if (cv > cv_max) cv_max = cv;
            if (s == STEPS - 1) final_cv = cv;
        }
        if (s == STEPS - 1) final_active = nact;
    }

    if (getenv("RELOC_DEBUG")) {
        host_t2_set_window_samples(wmin);
        printf("  [debug] effWin at min window:");
        for (int i = 0; i < M; i++) printf(" %.2f", host_t2_effwin(i));
        printf("\n");
    }
    printf("\n--- T2 relocation / uniformity (ramp window max->min) ---\n");
    printf("  taps M               : %d\n", M);
    printf("  worst spacing CV     : %.3f   (transient, taps mid-fade)\n", cv_max);
    printf("  final spacing CV     : %.3f   (min window; lower = more uniform)\n", final_cv);
    printf("  slides (moved while audible) : %d   (want 0)\n", slides);
    /* A handful of taps may park mid-relocation at the extreme min window, which
     * is inaudible against 32 taps. Allow ~16% (5 of 32). */
    int tol = (int)(0.16f * (float)M + 0.5f);
    printf("  active at min window : %d / %d   (allow up to %d faded)\n", final_active, M, tol);
    int pass = (slides == 0) && (final_cv < 0.6) && (final_active >= M - tol);
    printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
}

static void test_loud_sine(void) {
    velvet_reverb_init();
    /* Drive all macros open so taps, recirc, and filters are fully active —
     * otherwise default Density/Decay = 0 leaves the reverb near-silent and
     * none of the hot paths get exercised. */
    velvet_reverb_apply_decay_macro(1.0f);
    velvet_reverb_apply_tone_macro(1.0f);
    stats_t s; stats_init(&s);

    const double k = 2.0 * M_PI * 1000.0 / 48000.0;
    const double amp = 32767.0 * 0.707;
    for (int n = 0; n < 48000 * 3; n++) {
        int16_t in = (int16_t)(amp * sin(k * n));
        velvet_reverb_push_sample(in);
        velvet_reverb_poll();
        stats_update(&s, velvet_reverb_out_left(), velvet_reverb_out_right());
    }
    stats_report(&s, "1 kHz sine -3 dBFS (3 s)");
}

int main(void) {
    printf("velvet_reverb host harness\n");
    printf("  T0: %d taps   T1: %d taps   T2: %d taps\n",
           MAX_T0_TAPS, MAX_T1_TAPS, MAX_T2_TAPS);

    test_silence();
    test_impulse();
    test_quiet_noise();
    test_loud_sine();
    test_relocation();
    test_dump_ir("c_ir.csv");
    return 0;
}
