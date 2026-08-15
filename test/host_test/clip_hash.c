/* clip_hash.c — is a clipping stage generating broadband hash?
 *
 * THE MEASUREMENT
 * ---------------
 * The reverb is linear apart from its saturators and its modulators: the sparse
 * convolutions are FIR, so they cannot invent frequencies that were not fed in.
 * So if the input is band-limited to 2 kHz and energy still comes out at 6-11 kHz,
 * something created it. Clipping is the prime candidate, and at 24 kHz internally
 * it is doubly bad: harmonics of anything above 4 kHz land past Nyquist and fold
 * back down as broadband hash, which is heard as noise rather than as distortion.
 *
 * A sine is the wrong probe here and an earlier sweep with one was misleading —
 * it showed zero clipping at every level, because a sine has a low crest factor
 * while the reverb's dense tap sum turns broadband material into something far
 * peakier. Band-limited noise is what actually drives these stages into their
 * knees.
 *
 * WHAT IT REPORTS
 * ---------------
 * Every clipping point in the cascade, as a fraction of full scale, plus the
 * out-of-band energy that says whether the clipping is audible:
 *
 *   T0 in    predelay_out -> t0_ring   (soft_clip_int16)
 *   T0->T1   accT0 -> t1_ring          (soft_saturate_q15)
 *   T1->T2   accT1 -> t2_ring          (soft_saturate_q15)
 *   out      accL/R -> finalise        (soft_saturate_q15)
 *
 * Build with -DREVERB_HEADROOM=0.35f etc. to sweep the drive, and with
 * -DPREDELAY_INTERP_LINEAR to compare against the pre-allpass behaviour.
 *
 * Usage: ./clip_hash
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

extern float    host_dbg_t0_in_peak;   extern uint32_t host_dbg_t0_sat_count;
extern float    host_dbg_t01_peak;     extern uint32_t host_dbg_t01_sat;
extern float    host_dbg_t12_peak;     extern uint32_t host_dbg_t12_sat;
extern float    host_dbg_out_peak;     extern uint32_t host_dbg_out_sat;
extern float    host_dbg_final_peak;
extern uint64_t host_dbg_final_n, host_dbg_final_knee, host_dbg_final_over;
extern double   host_dbg_final_sumsq;

#define FS_OUT 48000.0
#define FFTN   8192

/* Mirrors the default in velvet_reverb.c, which is private to it. Pass the same
 * -DREVERB_HEADROOM to both to sweep. */
#ifndef REVERB_HEADROOM
#define REVERB_HEADROOM 0.5f
#endif

/* ---- FFT (size-parametric: also used to synthesise the probe) ---- */
static void fft_n(double *re, double *im, int N, int inverse)
{
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= N; len <<= 1) {
        double ang = (inverse ? 2.0 : -2.0) * M_PI / len;
        for (int i = 0; i < N; i += len)
            for (int k = 0; k < len / 2; k++) {
                double wr = cos(ang * k), wi = sin(ang * k);
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
                double vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
            }
    }
    if (inverse)
        for (int i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
}
static void fft_run(double *re, double *im) { fft_n(re, im, FFTN, 0); }

/* ---- The probe ----
 * White noise with an exact spectral hole, built in the frequency domain: noise
 * bins everywhere except NOTCH_LO..NOTCH_HI, which are set to zero. Two things
 * make this the right probe where the earlier 2 kHz-limited one was not. It still
 * carries full energy either side of the hole, so the loop fills with high
 * frequencies and the allpass change is actually exercised; and the hole is exact
 * rather than merely 100 dB down, so anything appearing inside it was manufactured.
 * Being synthesised as a periodic sequence it also loops seamlessly, so repeating
 * it for the length of the run adds no discontinuity of its own. */
#define PROBE_N   65536
#define NOTCH_LO  6000.0
#define NOTCH_HI  10000.0
#define MEAS_LO   7000.0
#define MEAS_HI    9000.0

static double probe[PROBE_N];

static void probe_build(void)
{
    static double re[PROBE_N], im[PROBE_N];
    uint32_t rng = 24601u;
    const double binhz = FS_OUT / PROBE_N;

    for (int b = 0; b < PROBE_N; b++) { re[b] = 0.0; im[b] = 0.0; }
    for (int b = 1; b < PROBE_N / 2; b++) {
        double f = b * binhz;
        if (f >= NOTCH_LO && f <= NOTCH_HI) continue;   /* the hole */
        if (f > 11900.0) continue;                      /* nothing the codec can carry */
        rng = rng * 1103515245u + 12345u;
        double ph = 2.0 * M_PI * ((double)((rng >> 8) & 0xFFFFFF) / 16777216.0);
        re[b] =  cos(ph); im[b] =  sin(ph);
        re[PROBE_N - b] =  cos(ph); im[PROBE_N - b] = -sin(ph);   /* conjugate */
    }
    fft_n(re, im, PROBE_N, 1);

    double pk = 0.0;
    for (int i = 0; i < PROBE_N; i++) if (fabs(re[i]) > pk) pk = fabs(re[i]);
    for (int i = 0; i < PROBE_N; i++) probe[i] = re[i] / pk;   /* peak-normalised */
}

/* Total energy in [lo, hi] Hz, in dB, Welch-averaged over the second half. */
static double band_energy_db(const float *sig, long n, double lo, double hi)
{
    static double re[FFTN], im[FFTN], acc[FFTN / 2];
    double win[FFTN], wsum = 0.0;
    for (int i = 0; i < FFTN; i++) {
        double t = (double)i / (FFTN - 1);
        win[i] = 0.35875 - 0.48829 * cos(2 * M_PI * t)
               + 0.14128 * cos(4 * M_PI * t) - 0.01168 * cos(6 * M_PI * t);
        wsum += win[i];
    }
    for (int b = 0; b < FFTN / 2; b++) acc[b] = 0.0;
    long segs = 0;
    for (long off = n / 2; off + FFTN <= n; off += FFTN / 2) {
        for (int i = 0; i < FFTN; i++) { re[i] = sig[off + i] * win[i]; im[i] = 0.0; }
        fft_run(re, im);
        for (int b = 0; b < FFTN / 2; b++) acc[b] += re[b] * re[b] + im[b] * im[b];
        segs++;
    }
    if (!segs) segs = 1;
    const double binhz = FS_OUT / FFTN;
    double e = 0.0;
    for (int b = 0; b < FFTN / 2; b++) {
        double f = b * binhz;
        if (f >= lo && f <= hi) e += acc[b] / segs / (wsum * wsum) * 4.0;
    }
    return 10.0 * log10(e + 1e-30);
}

/* ---- Steep input band limit: 4 cascaded 2-pole lowpasses at 2 kHz, giving
 * about -100 dB by 8 kHz, so the input itself contributes nothing to the band
 * being measured. ---- */
typedef struct { double b0, b1, b2, a1, a2, z1, z2; } LP;
static void lp_design(LP *f, double fc, double fs)
{
    double w = 2.0 * M_PI * fc / fs, al = sin(w) / (2.0 * 0.7071), a0 = 1.0 + al;
    double c = cos(w);
    f->b0 = (1.0 - c) / 2.0 / a0; f->b1 = (1.0 - c) / a0; f->b2 = f->b0;
    f->a1 = -2.0 * c / a0; f->a2 = (1.0 - al) / a0;
    f->z1 = f->z2 = 0.0;
}
static inline double lp_run(LP *f, double x)
{
    double y = f->b0 * x + f->z1;
    f->z1 = f->b1 * x - f->a1 * y + f->z2;
    f->z2 = f->b2 * x - f->a2 * y;
    return y;
}

/* The feedback ducker is a signal-dependent GAIN, not a waveshaper: duck_env
 * follows the input envelope with a 2 ms attack and multiplies the loop gain. A
 * gain that moves that fast is amplitude modulation at rates reaching into the
 * audio band, and its sidebands grow faster than linearly with level because the
 * duck depth itself grows with level. That makes it a candidate for spray that no
 * clipping counter would ever register. Set from main so it can be switched off. */
static float duck_amt = 1.0f;

int main(int argc, char **argv)
{
    const double secs = 12.0;
    long n = (long)(secs * FS_OUT);
    float *buf = malloc((size_t)n * sizeof(float));

    if (argc > 1) duck_amt = (float)atof(argv[1]);
    probe_build();

    printf("=== Clipping and the hash it makes ===\n\n");
    printf("Input is full-band noise with an exact hole from %.0f to %.0f Hz, so any\n",
           NOTCH_LO, NOTCH_HI);
    printf("output measured in %.0f-%.0f Hz was manufactured inside the reverb: the\n",
           MEAS_LO, MEAS_HI);
    printf("convolutions are FIR and cannot create frequencies. Energy remains on\n");
    printf("both sides of the hole, so the loop still fills with high frequencies —\n");
    printf("which a lowpassed probe would not do, and that is what a clipper needs\n");
    printf("to turn into hash, since above 4 kHz its harmonics land past the 12 kHz\n");
    printf("Nyquist and fold back down as broadband noise.\n\n");
    printf("Internal headroom = %.2f, decay 0.95, tone 1.0.\n\n", (double)REVERB_HEADROOM);

    printf("  %-9s %9s %14s %6s  %8s %8s   %s\n",
           "in dBFS", "notch", "out RMS", "peak", "in knee", "over", "post-makeup acc pk");

    const double levels[] = { -24.0, -18.0, -12.0, -6.0, -3.0, 0.0 };
    for (unsigned k = 0; k < sizeof levels / sizeof levels[0]; k++) {
        double amp = pow(10.0, levels[k] / 20.0);

        velvet_reverb_init();
        velvet_reverb_apply_decay_macro(0.95f);
        velvet_reverb_apply_tone_macro(1.0f);
        reverb_duck_amount = duck_amt;
        host_dbg_t0_in_peak = 0.0f; host_dbg_t0_sat_count = 0;
        host_dbg_t01_peak = 0.0f;   host_dbg_t01_sat = 0;
        host_dbg_t12_peak = 0.0f;   host_dbg_t12_sat = 0;
        host_dbg_out_peak = 0.0f;   host_dbg_out_sat = 0;
        host_dbg_final_peak = 0.0f; host_dbg_final_n = 0;
        host_dbg_final_knee = 0;    host_dbg_final_over = 0;
        host_dbg_final_sumsq = 0.0;

        for (long i = 0; i < n; i++) {
            double v = probe[i % PROBE_N] * amp;
            velvet_reverb_push_sample((int16_t)(v * 32000.0));
            if ((i % 32) == 31) velvet_reverb_poll();
            buf[i] = (float)velvet_reverb_out_left() / 32768.0f;
        }

        double lo = band_energy_db(buf, n, 100.0, 3000.0);
        double hi = band_energy_db(buf, n, MEAS_LO, MEAS_HI);
        double N  = (double)(host_dbg_final_n ? host_dbg_final_n : 1);

        printf("  %-9.1f %7.1f dB   %6.1f dBFS %6.2f  %7.3f%% %7.3f%%   %6.1f\n",
               levels[k], hi - lo,
               10.0 * log10(host_dbg_final_sumsq / N + 1e-30),
               (double)host_dbg_final_peak,
               100.0 * (double)host_dbg_final_knee / N,
               100.0 * (double)host_dbg_final_over / N,
               (double)host_dbg_out_peak * (1.0 / REVERB_HEADROOM));
    }

    printf("\n  If the over column is a small fraction of a percent then the hash comes\n");
    printf("  from occasional peaks, which a limiter can catch without any loss of\n");
    printf("  loudness. If it runs to whole percent the wet signal is simply louder\n");
    printf("  than the output can carry and something has to give.\n");
    free(buf);
    return 0;
}
