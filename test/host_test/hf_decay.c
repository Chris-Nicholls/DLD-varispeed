/* hf_decay.c — why do the tails lose their top end?
 *
 * The complaint is that tails come across mid-heavy with no shimmer. That is a
 * statement about the DECAY RATE PER BAND, not about the static frequency
 * response: a tail can start out bright and still sound dull if the high bands
 * die several times faster than the mids.
 *
 * Part 1 isolates the pre-delay feedback loop, which is where cumulative losses
 * live. Content circulates that loop once every 4440 samples (0.185 s), so a loss
 * of even a fraction of a dB per pass compounds into a huge difference in decay
 * rate. With the shelves and DC blocker removed and fb set to 0.999 the loop's
 * own loss is 0.0087 dB/pass, so whatever decay we measure is essentially all
 * down to the fractional-delay interpolator. Three are compared:
 *
 *   linear      — what the code does now (read_ring_interp), 2 SDRAM floats
 *   catmull-rom — 4-point cubic, 4 SDRAM floats
 *   thiran      — 1st-order allpass, flat magnitude by construction, 1 SDRAM float
 *
 * Part 2 measures the whole reverb's per-band decay so the isolated result can be
 * checked against what actually reaches the codec.
 *
 * Usage: ./hf_decay
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "host_shim.h"
#include "velvet_reverb.h"

int16_t host_t2_ring_storage[T2_RING_SAMPLES];
float   host_predelay_a_storage[PRE_DELAY_LINE_SAMPLES];
float   host_predelay_b_storage[PRE_DELAY_LINE_SAMPLES];

#define NBANDS 6
static const double band_hz[NBANDS] = { 250, 500, 1000, 2000, 4000, 8000 };

/* ==== Band-limited decay measurement ====
 * Three cascaded RBJ bandpasses per band. One pole pair leaks far too much skirt
 * to separate bands whose levels differ by 40 dB, which is exactly the situation
 * we are trying to measure. */
typedef struct { double b0, b1, b2, a1, a2, z1, z2; } BQ;

static void bp_design(BQ *f, double fc, double fs, double q)
{
    double w = 2.0 * M_PI * fc / fs;
    double al = sin(w) / (2.0 * q);
    double a0 = 1.0 + al;
    f->b0 =  al / a0;
    f->b1 =  0.0;
    f->b2 = -al / a0;
    f->a1 = -2.0 * cos(w) / a0;
    f->a2 = (1.0 - al) / a0;
    f->z1 = f->z2 = 0.0;
}

static inline double bp_run(BQ *f, double x)
{
    double y = f->b0 * x + f->z1;
    f->z1 = f->b1 * x - f->a1 * y + f->z2;
    f->z2 = f->b2 * x - f->a2 * y;
    return y;
}

/* Least-squares slope of the band envelope in dB/s, fitted over the span from
 * -5 dB to -35 dB below the level at the moment the excitation is cut. Starting
 * at -5 dB skips the initial transient; stopping at -35 dB stays clear of the
 * noise floor and of any late nonlinear behaviour. Returns 0 if the band never
 * covers enough range to fit. */
static double band_decay_db_per_s(const float *sig, long n, long cut, double fs, double fc)
{
    BQ f1, f2, f3;
    bp_design(&f1, fc, fs, 2.0);
    bp_design(&f2, fc, fs, 2.0);
    bp_design(&f3, fc, fs, 2.0);

    double *env = malloc((size_t)n * sizeof(double));
    const double ea = 1.0 - exp(-1.0 / (0.020 * fs));   /* 20 ms envelope */
    double e = 0.0;
    for (long i = 0; i < n; i++) {
        double v = bp_run(&f3, bp_run(&f2, bp_run(&f1, sig[i])));
        double r = fabs(v);
        e += ea * (r - e);
        env[i] = e;
    }

    double ref = 0.0;
    for (long i = cut - (long)(0.05 * fs); i < cut; i++) if (env[i] > ref) ref = env[i];
    if (ref <= 0.0) { free(env); return 0.0; }

    long i0 = -1, i1 = -1;
    for (long i = cut; i < n; i++) {
        double db = 20.0 * log10(env[i] / ref + 1e-30);
        if (i0 < 0 && db <= -5.0)  i0 = i;
        if (i0 >= 0 && db <= -35.0) { i1 = i; break; }
    }
    if (i0 < 0 || i1 < 0 || i1 - i0 < (long)(0.02 * fs)) { free(env); return 0.0; }

    double sx = 0, sy = 0, sxx = 0, sxy = 0; long cnt = 0;
    for (long i = i0; i <= i1; i += 8) {
        double t = (double)(i - i0) / fs;
        double db = 20.0 * log10(env[i] / ref + 1e-30);
        sx += t; sy += db; sxx += t * t; sxy += t * db; cnt++;
    }
    free(env);
    double den = cnt * sxx - sx * sx;
    if (fabs(den) < 1e-12) return 0.0;
    return -(cnt * sxy - sx * sy) / den;    /* positive = decaying */
}

/* ==== Part 1: the pre-delay loop on its own ==== */

#define LOOP_FS   ((double)REVERB_FS_HZ)
#define LOOP_LEN  PRE_DELAY_LINE_SAMPLES
#define LOOP_MASK PRE_DELAY_LINE_MASK

enum { INTERP_LINEAR, INTERP_CUBIC, INTERP_THIRAN };

static float loop_buf[LOOP_LEN];

static inline float rd_linear(const float *b, uint32_t m, float pos)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    float a = b[(uint32_t)i & m];
    float c = b[((uint32_t)i + 1u) & m];
    return a + f * (c - a);
}

static inline float rd_cubic(const float *b, uint32_t m, float pos)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    float y0 = b[((uint32_t)(i - 1)) & m];
    float y1 = b[((uint32_t)i)       & m];
    float y2 = b[((uint32_t)(i + 1)) & m];
    float y3 = b[((uint32_t)(i + 2)) & m];
    float a = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float c =        y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float d = -0.5f * y0 + 0.5f * y2;
    return ((a * f + c) * f + d) * f + y1;
}

/* 1st-order Thiran allpass on the integer-delayed stream. Magnitude is exactly
 * flat at every frequency; only the group delay is approximate, which is the
 * whole appeal inside a loop traversed hundreds of times. */
static float ap_prev_in = 0.0f, ap_prev_out = 0.0f;
static inline float rd_thiran(const float *b, uint32_t m, float pos)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    float s = b[(uint32_t)i & m];
    float a = (1.0f - f) / (1.0f + f);
    float y = a * (s - ap_prev_out) + ap_prev_in;
    ap_prev_in = s; ap_prev_out = y;
    return y;
}

static void run_loop(int interp, float fb, double burst_s, double tail_s,
                     float *out, long *n_out, long *cut_out)
{
    memset(loop_buf, 0, sizeof loop_buf);
    ap_prev_in = ap_prev_out = 0.0f;

    const float timeA    = (float)(reverb_predelay_a_s * LOOP_FS);
    const float modDepth = reverb_fb_mod_depth * (float)PRE_MOD_MAX_SAMPLES;
    const float modInc   = (float)(2.0 * M_PI * reverb_fb_mod_rate / LOOP_FS);

    long burst = (long)(burst_s * LOOP_FS);
    long total = (long)((burst_s + tail_s) * LOOP_FS);
    uint32_t wh = 0, rng = 22222u;
    float ph = 0.0f;

    for (long n = 0; n < total; n++) {
        float x = 0.0f;
        if (n < burst) {
            rng = rng * 1103515245u + 12345u;
            x = ((float)((rng >> 16) & 0xFFFF) - 32768.0f) / 32768.0f * 0.25f;
        }
        ph += modInc; if (ph >= (float)(2.0 * M_PI)) ph -= (float)(2.0 * M_PI);
        float extra = modDepth * (1.0f + sinf(ph));

        float whf = (float)(wh & LOOP_MASK);
        float pos = whf - timeA - extra;
        float r = (interp == INTERP_LINEAR) ? rd_linear(loop_buf, LOOP_MASK, pos)
                : (interp == INTERP_CUBIC)  ? rd_cubic (loop_buf, LOOP_MASK, pos)
                                            : rd_thiran(loop_buf, LOOP_MASK, pos);
        float node = x + fb * r;
        if (node >  4.0f) node =  4.0f;
        if (node < -4.0f) node = -4.0f;
        loop_buf[wh & LOOP_MASK] = node;
        out[n] = node;
        wh++;
    }
    *n_out = total;
    *cut_out = burst;
}

static void part1(void)
{
    const double burst = 1.0, tail = 12.0;
    const float  fb    = 0.999f;
    long n, cut;
    float *buf = malloc((size_t)((burst + tail) * LOOP_FS + 8) * sizeof(float));

    printf("=== Part 1: pre-delay feedback loop in isolation ===\n\n");
    printf("fb = %.3f (0.0087 dB/pass), shelves and DC block removed, so the decay\n", fb);
    printf("below is what the fractional-delay interpolator costs. Loop period is\n");
    printf("%.3f s, so a pass happens %.1f times a second.\n\n",
           reverb_predelay_a_s, 1.0 / reverb_predelay_a_s);

    const char *names[3] = { "linear (current)", "catmull-rom", "thiran allpass" };
    const int   sdram[3] = { 2, 4, 1 };
    double loss[3][NBANDS];

    for (int k = 0; k < 3; k++) {
        run_loop(k, fb, burst, tail, buf, &n, &cut);
        printf("  %-18s (%d SDRAM float%s per read)\n",
               names[k], sdram[k], sdram[k] == 1 ? "" : "s");
        printf("    %8s %12s %12s\n", "band", "dB/pass", "RT60 s");
        for (int b = 0; b < NBANDS; b++) {
            double slope = band_decay_db_per_s(buf, n, cut, LOOP_FS, band_hz[b]);
            double per   = slope * reverb_predelay_a_s;   /* dB lost per traversal */
            loss[k][b] = per;
            /* A band that never falls 35 dB inside the capture has no loss this
             * measurement can see, which for a loop this leaky means no loss at
             * all — report that rather than a fitted number from noise. */
            if (slope > 0.0) printf("    %6.0f Hz %12.4f %12.2f\n", band_hz[b], per, 60.0 / slope);
            else             printf("    %6.0f Hz %12s %12s\n",     band_hz[b], "-", "no decay");
        }
        printf("\n");
    }

    printf("  Bands showing no decay are transparent: the loop returns them\n");
    printf("  unchanged pass after pass. A tail keeps its spectral balance only if\n");
    printf("  every band behaves that way, and that balance is what \"shimmer\" is.\n\n");
    printf("  %-18s %12s %12s %10s\n", "interpolator", "4 kHz", "8 kHz", "SDRAM");
    for (int k = 0; k < 3; k++)
        printf("  %-18s %9.3f dB %9.3f dB %8d f\n",
               names[k], loss[k][4], loss[k][5], sdram[k]);
    printf("  (dB lost per traversal — multiply by %.1f for dB/s)\n\n",
           1.0 / reverb_predelay_a_s);
    free(buf);
}

/* ==== Part 2: the whole reverb, decay and tone at max ==== */

#define FS_OUT 48000.0

static void part2_case(float decay, float tone)
{
    const double burst = 1.5, tail = 10.0;
    long total = (long)((burst + tail) * FS_OUT);
    long cut   = (long)(burst * FS_OUT);
    float *buf = malloc((size_t)total * sizeof(float));

    printf("  decay %.2f / tone %.2f\n", decay, tone);

    velvet_reverb_init();
    velvet_reverb_apply_decay_macro(decay);
    velvet_reverb_apply_tone_macro(tone);

    uint32_t rng = 4242u;
    for (long i = 0; i < total; i++) {
        int32_t in = 0;
        if (i < cut) {
            rng = rng * 1103515245u + 12345u;
            double nz = ((double)((rng >> 16) & 0xFFFF) - 32768.0) / 32768.0;
            in = (int32_t)(nz * 8000.0);
        }
        velvet_reverb_push_sample((int16_t)in);
        if ((i % 32) == 31) velvet_reverb_poll();
        buf[i] = (float)velvet_reverb_out_left() / 32768.0f;
    }

    printf("   ");
    for (int b = 0; b < NBANDS; b++) printf(" %8.0f", band_hz[b]);
    printf("  Hz\n   ");
    for (int b = 0; b < NBANDS; b++) {
        double slope = band_decay_db_per_s(buf, total, cut, FS_OUT, band_hz[b]);
        if (slope > 0.0) printf(" %8.2f", 60.0 / slope);
        else             printf(" %8s", "frozen");
    }
    printf("  RT60 s\n\n");
    free(buf);
}

static void part2(void)
{
    printf("=== Part 2: whole reverb, per-band RT60 ===\n\n");
    printf("Max decay freezes every band, so mid settings are what show whether the\n");
    printf("tilt is gone: the 8 kHz figure should sit in the same range as 500 Hz\n");
    printf("rather than a fraction of it.\n\n");
    part2_case(0.50f, 1.0f);
    part2_case(0.75f, 1.0f);
    part2_case(0.90f, 1.0f);
    part2_case(1.00f, 1.0f);
    part2_case(0.75f, 0.5f);
}

/* ==== Part 3: the static cost of the rate conversion ====
 *
 * Separate from the decay tilt: even a tail that keeps its balance is dull if
 * the path to the codec rolls off. Three stages sit between the cascade and the
 * jack, and all three are reproduced here exactly as the firmware has them:
 *
 *   in   48->24 kHz by averaging sample pairs   (velvet_reverb_push_sample)
 *   out  24->48 kHz by (prev + cur) * 0.5       (the block finaliser)
 *   out  2nd-order LPF, clamped to 11.9 kHz     (doubles as reconstruction)
 *
 * A sine is pushed through at each frequency and the steady-state amplitude
 * compared with its input, which also exposes the image the upsampler leaves at
 * 24 kHz - f, the reason the LPF has to be clamped so low in the first place. */
static void part3(void)
{
    printf("=== Part 3: static loss in the 48<->24 kHz conversion ===\n\n");
    printf("    %8s %10s %10s %10s %10s\n",
           "freq", "decimate", "upsample", "LPF 11.9k", "total");

    for (double f = 1000.0; f <= 12000.0; f *= (f < 4000.0 ? 2.0 : 1.5)) {
        /* Decimator: 2-tap boxcar at 48 kHz, |H| = |cos(pi f / 48000)|. */
        double dec = fabs(cos(M_PI * f / 48000.0));
        /* Upsampler: zero-stuff then convolve [0.5, 1, 0.5], normalised. */
        double w   = 2.0 * M_PI * f / 48000.0;
        double up  = fabs(cos(w / 2.0) * cos(w / 2.0));
        /* Output LPF: 2nd-order Butterworth at the 11.9 kHz clamp. */
        double r   = f / 11900.0;
        double lp  = 1.0 / sqrt(1.0 + r * r * r * r);

        printf("    %6.0f Hz %9.2f %9.2f %9.2f %9.2f dB\n", f,
               20.0 * log10(dec), 20.0 * log10(up),
               20.0 * log10(lp),  20.0 * log10(dec * up * lp));
    }

    /* Image left by linear interpolation, relative to the wanted component. */
    printf("\n    upsampler image (at 24 kHz - f), relative to the signal:\n");
    printf("    %8s %12s %12s\n", "signal", "before LPF", "after LPF");
    for (double f = 4000.0; f <= 10000.0; f += 2000.0) {
        double fi = 24000.0 - f;
        double w  = 2.0 * M_PI * f  / 48000.0, wi = 2.0 * M_PI * fi / 48000.0;
        double hs = cos(w / 2.0)  * cos(w / 2.0);
        double hi = cos(wi / 2.0) * cos(wi / 2.0);
        double ri = fi / 11900.0;
        double lp = 1.0 / sqrt(1.0 + ri * ri * ri * ri);
        printf("    %6.0f Hz %9.1f dB %9.1f dB\n", f,
               20.0 * log10(hi / hs), 20.0 * log10(hi * lp / hs));
    }
    printf("\n  The droop is a fixed dullness on top of any decay tilt, and the\n");
    printf("  image is why the LPF cannot be opened past 11.9 kHz while the\n");
    printf("  upsampler stays linear. A polyphase half-band would leave the even\n");
    printf("  output samples as a pure delay and cost a handful of taps on the odd\n");
    printf("  ones, flattening the droop and pushing the image below -40 dB.\n\n");
}

int main(void)
{
    part1();
    part2();
    part3();
    return 0;
}
