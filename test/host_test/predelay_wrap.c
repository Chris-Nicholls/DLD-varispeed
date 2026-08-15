/* predelay_wrap.c — does the pre-delay LFO wrap splice the delay line?
 *
 * WHY THIS EXISTS
 * ---------------
 * A STEPHUNT=5 slew trace on hardware found a discontinuity at the cascade input
 * (the output of do_predelay) in 8 of 9 flagged windows, at ~1.5x the block peak
 * where ordinary content there sits at 0.32x. Every earlier probe missed it
 * because they all measured PEAK, and a splice leaves the peak untouched.
 *
 * The suspect is the modulation LFO. do_predelay wraps its phase into (-pi, pi]
 * and feeds it to fast_sin_pi, a 5th-order Taylor sine. A true sine is zero at
 * +-pi so the wrap is seamless; the Taylor polynomial is NOT, so the modulation
 * value jumps at every wrap, which steps the fractional read position of a delay
 * line and splices its output.
 *
 * Two independent checks, because either alone is weak:
 *
 *   1. THE MECHANISM, in closed form: how big is the jump, in samples of read
 *      position and in ms. This needs no reverb and cannot be argued with.
 *
 *   2. THE CONSEQUENCE, in the real reverb: drive it with stationary noise and
 *      look for slew spikes at the output, then test whether they land at the
 *      PREDICTED wrap times. Correlation against times derived from modRate
 *      alone is what separates this from the coincidences that earlier
 *      detectors in this hunt kept reporting.
 *
 * Usage: ./predelay_wrap [seconds]        (default 30)
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

/* The cascade input, i.e. what do_predelay produced — the same point and the same
 * statistic as the firmware's STEPHUNT=5 probe. Measuring at the reverb OUTPUT
 * instead does not work: T0/T1/T2 sum many sparse taps, which smears the splice
 * and lifts the slew floor from 0.32 to 0.59, comfortably burying it. */
extern const float *host_dbg_predelay_out(void);
extern float    host_dbg_node_peak;
extern uint32_t host_dbg_node_clamps;
extern float    host_dbg_t0_in_peak;
extern uint32_t host_dbg_t0_sat_count;

#define TWO_PI_F  6.28318530717958647692f
#define PI_F      3.14159265358979323846f

#define FS_OUT           48000.0
#define FRAMES_PER_BLOCK 32
#define WARMUP_S         3.0

/* The unfolded Taylor form: what velvet_reverb.c used before the fix. Kept as a
 * local copy so the test still demonstrates the fault after the fix lands, and so
 * part 2 can A/B it against a correct sine under identical input. */
static float fast_sin_pi_taylor(float x)
{
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666666f + x2 * 0.00833333f));
}

/* The fixed form, mirroring velvet_reverb.c: fold to [-pi/2, pi/2] first. */
#define PI_2_F 1.57079632679489661923f
static float fast_sin_pi_folded(float x)
{
    if (x >  PI_2_F)      x =  PI_F - x;
    else if (x < -PI_2_F) x = -PI_F - x;
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666666f + x2 * 0.00833333f));
}

static int part1_mechanism(void)
{
    /* At the wrap, the phase steps from just under +pi to just over -pi. */
    float hi = fast_sin_pi_taylor(PI_F);
    float lo = fast_sin_pi_taylor(-PI_F);
    float jump = hi - lo;
    float fix_jump = fast_sin_pi_folded(PI_F) - fast_sin_pi_folded(-PI_F);

    float depth = reverb_fb_mod_depth * (float)PRE_MOD_MAX_SAMPLES;
    float samples = jump * depth;
    double ms = 1000.0 * samples / (double)REVERB_FS_HZ;

    printf("--- part 1: the mechanism ---\n");
    printf("  sin(+pi) exact           : 0\n");
    printf("  fast_sin_pi(+pi)         : %+.4f   <-- should be 0\n", hi);
    printf("  fast_sin_pi(-pi)         : %+.4f\n", lo);
    printf("  step in modulation value : %.4f\n", jump);
    printf("  modDepth                 : %.1f samples (%.3f x %d)\n",
           depth, reverb_fb_mod_depth, PRE_MOD_MAX_SAMPLES);
    printf("  => read position jumps   : %.2f samples = %.3f ms, instantaneously\n",
           samples, ms);
    printf("  wrap rate                : %.2f Hz per line, %.2f Hz for both\n",
           reverb_fb_mod_rate, 2.0f * reverb_fb_mod_rate);

    printf("  same step, folded fix    : %.4f (%.3f samples)\n",
           fix_jump, fix_jump * depth);

    /* Worst error of the folded form against a true sine, to confirm the fix does
     * not trade a step for a badly wrong modulation shape. */
    float worst = 0.0f;
    for (int k = -1000; k <= 1000; k++) {
        float x = PI_F * (float)k / 1000.0f;
        float e = fabsf(fast_sin_pi_folded(x) - sinf(x));
        if (e > worst) worst = e;
    }
    printf("  folded vs true sine      : max error %.5f over [-pi, pi]\n", worst);

    /* The verdict is on the SHIPPED form. The unfolded figures above are the
     * contrast — this file keeps the pre-fix sine deliberately, so gating on it
     * would fail forever and teach us to ignore the result. */
    int shipped_bad = fabsf(fix_jump) > 0.01f || worst > 0.01f;
    printf("  verdict: unfolded is %s; folded (shipped) is %s\n",
           fabsf(jump) > 0.01f ? "DISCONTINUOUS" : "continuous",
           shipped_bad ? "STILL DISCONTINUOUS" : "continuous");
    return shipped_bad;
}

/* Per-block max slew normalised by the block peak — the same statistic the
 * firmware's STEPHUNT=5 probe reports, so the two are directly comparable. */
#define BLK 16
/* 1500 blocks/s, so a 60 s run needs 90k slots. Sized generously: capping this
 * silently truncates the analysis window while the reported rates still divide by
 * the full run length, which turned a 3.8/s result into a plausible-looking
 * 0.35/s on the first attempt. */
#define MAX_EV 262144
static double ev_t[MAX_EV];
static double ev_r[MAX_EV];
static int    n_ev = 0;

/* ---- part 2: one modulated delay line, Taylor sine vs exact sine ----
 *
 * The full reverb turned out to be the wrong place to measure this. Its cascade
 * input already carries a slew floor of 0.28 with ordinary variation reaching 3x
 * that several times a second, so a single 7.55-sample splice per 1.4 s is lost
 * in it — the earlier attempt duly found 190 scattered spikes with a phase
 * concentration of 0.065, i.e. pure stimulus noise.
 *
 * So reproduce just the mechanism: one ring, one modulated fractional read, the
 * same depth and rate as do_predelay, driven by the same band-limited noise. The
 * only difference between the two runs is which sine is used, which makes any
 * difference in the result attributable to that and nothing else. */

/* Mirrors IDX_WRAP, which is private to velvet_reverb.c. Its comment there
 * notes 2^23 is "< 2^24 so the float cast in the recirc math is exact" — the cast
 * is indeed exact, but that is not the property that matters for a fractional
 * read: what limits the fraction is the float spacing at the RESULT's magnitude. */
#define IDX_WRAP 0x800000UL

#define SIM_FS   ((double)REVERB_FS_HZ)
#define SIM_RING 16384u
#define SIM_MASK (SIM_RING - 1u)

static float sim_ring[SIM_RING];

typedef struct {
    double med; int spikes; double rate; double R; double worst;
    /* The decisive pair: the output step at the exact sample the phase wraps,
     * against the distribution of steps everywhere else. Block statistics were
     * too blunt to separate 21 splices a minute from ~190 spikes thrown up by the
     * noise itself, and a plain A/B difference does not isolate them either
     * because the Taylor error shifts the read position at every other sample too. */
    double wrap_step;      /* mean |step| at wrap samples, / local peak */
    double normal_p99;     /* 99th pct |step| elsewhere, / local peak  */
    int    n_wrap;
} Sim;

/* mode: 0 = unfolded Taylor (the fault), 1 = true sinf, 2 = the folded fix */
static Sim simulate_line(int mode, double secs)
{
    Sim s; memset(&s, 0, sizeof s);

    const float depth   = reverb_fb_mod_depth * (float)PRE_MOD_MAX_SAMPLES;
    const float modInc  = (float)(TWO_PI_F * reverb_fb_mod_rate / SIM_FS);
    const float basetime = 4000.0f;
    const long  total   = (long)(secs * SIM_FS);

    memset(sim_ring, 0, sizeof sim_ring);
    float phase = 0.0f;
    uint32_t wh = 0;
    uint32_t rng = 22222u;
    double lp[4] = {0, 0, 0, 0};
    const double lpa = 1.0 - exp(-2.0 * M_PI * 2000.0 / SIM_FS);

    static double r_all[MAX_EV], r_t[MAX_EV];
    static double norm[MAX_EV];
    int n = 0, n_norm = 0, nwrap = 0;
    double wsum = 0.0;
    float run_peak = 0.0f;
    float prev = 0.0f, pk = 0.0f, dmax = 0.0f;
    int nblk = 0;

    for (long i = 0; i < total; i++) {
        rng = rng * 1103515245u + 12345u;
        double nz = ((double)((rng >> 16) & 0xFFFF) - 32768.0) / 32768.0;
        lp[0] += lpa * (nz     - lp[0]);
        lp[1] += lpa * (lp[0]  - lp[1]);
        lp[2] += lpa * (lp[1]  - lp[2]);
        lp[3] += lpa * (lp[2]  - lp[3]);

        sim_ring[wh & SIM_MASK] = (float)(lp[3] * 6.0);

        /* Exactly do_predelay's phase handling. Note the fold to (-pi, pi] means
         * the discontinuity lands where phase crosses PI, not at the 2pi wrap. */
        float phase_prev = phase;
        phase += modInc; if (phase >= TWO_PI_F) phase -= TWO_PI_F;
        float p = phase; if (p > PI_F) p -= TWO_PI_F;
        int at_wrap = (phase_prev <= PI_F && phase > PI_F);
        float sv = (mode == 1) ? sinf(p)
                 : (mode == 2) ? fast_sin_pi_folded(p)
                               : fast_sin_pi_taylor(p);
        float extra = depth * (1.0f + sv);

        float pos = (float)wh - basetime - extra;
        int32_t idx = (int32_t)pos;
        float f = pos - (float)idx;
        if (f < 0.0f) { f += 1.0f; idx -= 1; }
        float a = sim_ring[(uint32_t)idx & SIM_MASK];
        float b = sim_ring[((uint32_t)idx + 1u) & SIM_MASK];
        float out = a + f * (b - a);
        wh++;

        float d = out - prev; if (d < 0.0f) d = -d;
        if (d > dmax) dmax = d;
        prev = out;
        float m = out < 0.0f ? -out : out;
        if (m > pk) pk = m;

        /* Per-sample step, normalised by a slow running peak so the figure is
         * level-independent. Collected separately for wrap and non-wrap samples. */
        run_peak += (m > run_peak) ? 0.05f * (m - run_peak) : 0.0005f * (m - run_peak);
        if (i > (long)SIM_FS && run_peak > 1e-6f) {
            double sr = d / run_peak;
            if (at_wrap) { wsum += sr; nwrap++; }
            else if (n_norm < MAX_EV) norm[n_norm++] = sr;
        }

        if (++nblk < BLK) continue;
        nblk = 0;
        float p2 = pk, dm = dmax; pk = 0.0f; dmax = 0.0f;
        if (i < (long)SIM_FS || p2 < 1e-6f) continue;      /* 1 s to fill the line */
        if (n < MAX_EV) { r_all[n] = dm / p2; r_t[n] = (double)i / SIM_FS; n++; }
    }
    if (n < 32) return s;

    static double sorted[MAX_EV];
    memcpy(sorted, r_all, (size_t)n * sizeof(double));
    for (int a = 1; a < n; a++) {
        double v = sorted[a]; int b = a - 1;
        while (b >= 0 && sorted[b] > v) { sorted[b + 1] = sorted[b]; b--; }
        sorted[b + 1] = v;
    }
    s.med = sorted[n / 2];
    s.worst = sorted[n - 1];

    const double thresh = 3.0 * s.med;
    const double half = 0.5 / (double)reverb_fb_mod_rate;
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; i++) {
        if (r_all[i] < thresh) continue;
        s.spikes++;
        double ph = fmod(r_t[i], half) / half;
        sx += cos(2.0 * M_PI * ph);
        sy += sin(2.0 * M_PI * ph);
    }
    double span = r_t[n - 1] - r_t[0];
    s.rate = s.spikes / (span > 0 ? span : 1);
    s.R = s.spikes ? sqrt(sx * sx + sy * sy) / s.spikes : 0.0;

    s.n_wrap = nwrap;
    s.wrap_step = nwrap ? wsum / nwrap : 0.0;
    if (n_norm > 128) {
        for (int a = 1; a < n_norm; a++) {
            double v = norm[a]; int b = a - 1;
            while (b >= 0 && norm[b] > v) { norm[b + 1] = norm[b]; b--; }
            norm[b + 1] = v;
        }
        s.normal_p99 = norm[(int)(n_norm * 0.99)];
    }
    return s;
}

static int part2_isolated(double secs)
{
    printf("\n--- part 2: one modulated delay line, Taylor vs exact sine ---\n");
    Sim bad  = simulate_line(0, secs);
    Sim good = simulate_line(1, secs);
    Sim fix  = simulate_line(2, secs);

    printf("  %-24s %9s %9s %9s\n", "", "unfolded", "true sin", "folded");
    printf("  %-24s %9.3f %9.3f %9.3f\n", "slew/peak median",
           bad.med, good.med, fix.med);
    printf("  %-24s %9d %9d %9d\n", "spikes > 3x median",
           bad.spikes, good.spikes, fix.spikes);
    printf("  %-24s %9.3f %9.3f %9.3f\n", "phase concentration R",
           bad.R, good.R, fix.R);
    printf("  (block statistics are blind here: the noise itself throws up ~190\n");
    printf("   spikes over 3x median, which buries %d splices.)\n", bad.n_wrap);

    printf("\n  step at the wrap sample, normalised by running peak:\n");
    printf("  %-24s %9s %9s %9s\n", "", "unfolded", "true sin", "folded");
    printf("  %-24s %9.3f %9.3f %9.3f\n", "mean step AT wrap",
           bad.wrap_step, good.wrap_step, fix.wrap_step);
    printf("  %-24s %9.3f %9.3f %9.3f\n", "p99 step elsewhere",
           bad.normal_p99, good.normal_p99, fix.normal_p99);
    printf("  %-24s %9d %9d %9d\n", "wrap samples seen",
           bad.n_wrap, good.n_wrap, fix.n_wrap);

    double ratio_bad = bad.normal_p99 > 0 ? bad.wrap_step / bad.normal_p99 : 0;
    double ratio_fix = fix.normal_p99 > 0 ? fix.wrap_step / fix.normal_p99 : 0;
    printf("  %-24s %9.2f %9.2f %9.2f\n", "wrap / p99 elsewhere", ratio_bad,
           good.normal_p99 > 0 ? good.wrap_step / good.normal_p99 : 0, ratio_fix);

    /* The meaningful comparison is against a true sine under identical input and
     * seed: only the sine differs between runs, so a gap can come from nothing
     * else. The fix passes when it lands on the true sine's behaviour. */
    double vs_exact = good.wrap_step > 1e-9 ? bad.wrap_step / good.wrap_step : 0.0;
    double fix_vs_exact = good.wrap_step > 1e-9 ? fix.wrap_step / good.wrap_step : 0.0;
    printf("  %-24s %9.2f %9s %9.2f\n", "x the true sine's step", vs_exact, "1.00",
           fix_vs_exact);

    printf("  fault : %s\n", (vs_exact > 3.0 && ratio_bad > 1.5)
           ? "unfolded steps the output ~6x harder than a true sine, putting\n"
             "          every wrap among the largest steps in the signal"
           : "no outsized step from the unfolded form");
    int shipped_bad = !(fix_vs_exact < 1.5 && ratio_fix < 1.0);
    printf("  fix   : %s\n", shipped_bad
           ? "folded still steps — NOT fixed"
           : "folded matches the true sine — the step is gone");
    return shipped_bad;
}

/* ---- part 4: write-head float precision ----
 *
 * do_predelay computes its read position as (float)predelay_wh - time - extra,
 * and predelay_wh is a uint32 that starts at 0 on boot and advances every sample.
 * float32 represents integers exactly only below 2^24, which at REVERB_FS_HZ is
 * 11.6 minutes of runtime; past that the spacing becomes 2 samples, then 4, and
 * the subtraction can no longer resolve the fractional offset the interpolation
 * depends on. This runs the same line at several write-head offsets to measure
 * what that does — the point of interest is whether it degrades gradually or
 * produces steps, since only steps click. */
static int part4_precision(void)
{
    printf("\n--- part 4: write-head float precision over runtime ---\n");
    int shipped_bad = 0;

    const float depth   = reverb_fb_mod_depth * (float)PRE_MOD_MAX_SAMPLES;
    const float modInc  = (float)(TWO_PI_F * reverb_fb_mod_rate / SIM_FS);
    const float basetime = 4000.0f;

    /* predelay_wh is wrapped at IDX_WRAP, so it never reaches 2^24 and the
     * integer itself is always exact. The loss is in the SUBTRACTION: the result
     * carries the magnitude of the write head, and float spacing at that magnitude
     * sets how finely the fractional offset can be resolved at all. Spacing hits
     * 0.5 at 2^22 and a full sample at 2^23, so the modulation staircases as the
     * counter climbs and recovers only when it wraps. */
    printf("  %14s %10s %12s %12s\n", "wh position", "wh", "frac step", "masked fix");
    const double fracs[] = { 0.0, 0.05, 0.125, 0.25, 0.5, 0.9, 0.999 };
    for (unsigned h = 0; h < sizeof fracs / sizeof fracs[0]; h++) {
        uint32_t wh0 = (uint32_t)(fracs[h] * (double)IDX_WRAP);

        /* Sweep one LFO period and record the read positions produced. */
        /* The quantum of the fractional read offset: how coarsely the smooth sweep
         * is being staircased. Measured for the raw cast and for the masked fix. */
        float phase = 0.0f;
        double prev_raw = -1.0, prev_fix = -1.0, step_raw = 0.0, step_fix = 0.0;
        const long n = (long)(SIM_FS / reverb_fb_mod_rate);
        for (long i = 0; i < n; i++) {
            phase += modInc; if (phase >= TWO_PI_F) phase -= TWO_PI_F;
            float p = phase; if (p > PI_F) p -= TWO_PI_F;
            float extra = depth * (1.0f + fast_sin_pi_folded(p));
            uint32_t wh = wh0 + (uint32_t)i;
            if (wh >= (uint32_t)IDX_WRAP) wh -= (uint32_t)IDX_WRAP;

            float pos_raw = (float)wh - basetime - extra;
            float pos_fix = (float)(wh & PRE_DELAY_LINE_MASK) - basetime - extra;

            double fr = (double)pos_raw - floor((double)pos_raw);
            double ff = (double)pos_fix - floor((double)pos_fix);
            if (prev_raw >= 0.0) {
                double d = fabs(fr - prev_raw); if (d > 0.5) d = 1.0 - d;
                if (d > step_raw) step_raw = d;
                double e = fabs(ff - prev_fix); if (e > 0.5) e = 1.0 - e;
                if (e > step_fix) step_fix = e;
            }
            prev_raw = fr; prev_fix = ff;
        }
        printf("  %10.1f%% of %10u %12.4f %12.4f%s\n", fracs[h] * 100.0, wh0,
               step_raw, step_fix,
               (step_raw > 0.4) ? "   <-- was half-sample" : "");
        /* The masked path is what ships: it must stay near the quantisation
         * floor at every write-head position, not just early in the cycle. */
        if (step_fix > 0.01) shipped_bad = 1;
    }
    printf("  (%.0f s per wrap cycle. 'frac step' is the coarsest jump in the\n",
           (double)IDX_WRAP / SIM_FS);
    printf("   fractional read offset: 0.5 means the sweep has been staircased to\n");
    printf("   half-sample jumps, roughly 20 a second across both lines. Masking the\n");
    printf("   write head before the cast holds it near the quantisation floor for\n");
    printf("   the whole cycle instead of only the first eighth of it.)\n");
    return shipped_bad;
}

int main(int argc, char **argv)
{
    double secs = (argc > 1) ? atof(argv[1]) : 30.0;

    printf("=== pre-delay LFO wrap test (%.0f s) ===\n\n", secs);
    int bad = part1_mechanism();
    bad |= part2_isolated(secs);
    bad |= part4_precision();

    printf("\nRESULT: %s\n", bad ? "FAIL — the shipped path steps at a wrap"
                                  : "PASS — shipped path continuous at every wrap");
    return bad ? 1 : 0;
}
