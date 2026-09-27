/* oct_pitch.c — does the reverb's octave shift land where it should?
 *
 * Feeds noise band-limited to 350-500 Hz into the reverb with the octave set
 * to off, up and down, and measures the output level in that band and in the
 * bands an octave below and above (Goertzel, 2 Hz steps, over the last few
 * seconds once the tail has built up). With the shift on, the octave band
 * should come up to within a few dB of the source band (it is an equal blend),
 * and well above where it sits with the shift off.
 *
 * Noise rather than a sine: a steady sine rings the pre-delay loop's comb
 * resonances at one frequency, so its level says more about where it lands on
 * the comb than about the shift.
 *
 * Then a switching run flips the octave every 150 ms while a sine plays, to
 * check the crossfade keeps the output finite and unclipped.
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

#define FS_OUT           48000.0
#define FRAMES_PER_BLOCK 32

static double goertzel(const float *x, long n, double f)
{
    double w = 2.0 * M_PI * f / FS_OUT, c = 2.0 * cos(w), s1 = 0, s2 = 0;
    for (long i = 0; i < n; i++) {
        double s0 = x[i] + c * s1 - s2;
        s2 = s1; s1 = s0;
    }
    return sqrt(s1 * s1 + s2 * s2 - c * s1 * s2) / (double)n;
}

/* RMS level across [lo, hi] Hz */
static double band(const float *x, long n, double lo, double hi)
{
    double e = 0;
    for (double g = lo; g <= hi; g += 2.0) {
        double a = goertzel(x, n, g);
        e += a * a;
    }
    return sqrt(e);
}

typedef struct { double f220, f440, f880; long sat, nonfinite; } Res;

static Res run(int8_t octave, double secs, int switching)
{
    Res r; memset(&r, 0, sizeof r);
    velvet_reverb_init();
    velvet_reverb_apply_decay_macro(0.5f);
    velvet_reverb_apply_tone_macro(0.7f);
    velvet_reverb_set_octave(octave);

    const long total = (long)(secs * FS_OUT);
    const long meas  = (long)(3.0 * FS_OUT);
    float *out = malloc(sizeof(float) * (size_t)meas);
    int8_t oct = octave;

    /* 350-500 Hz: a 4th-order bandpass (two RBJ constant-peak biquads) on
     * white noise */
    const double f0 = sqrt(350.0 * 500.0), bw = log2(500.0 / 350.0);
    const double w0 = 2.0 * M_PI * f0 / FS_OUT;
    const double al = sin(w0) * sinh(log(2.0) / 2.0 * bw * w0 / sin(w0));
    const double a0 = 1.0 + al;
    const double b0 = al / a0, b2 = -al / a0, a1 = -2.0 * cos(w0) / a0, a2 = (1.0 - al) / a0;
    double z[2][2] = {{0, 0}, {0, 0}};
    uint32_t rng = 777u;

    for (long i = 0; i < total; i++) {
        rng = rng * 1103515245u + 12345u;
        double v = ((double)((rng >> 16) & 0xFFFF) - 32768.0) / 32768.0;
        for (int k = 0; k < 2; k++) {
            double y = b0 * v + z[k][0];
            z[k][0] = -a1 * y + z[k][1];
            z[k][1] = b2 * v - a2 * y;
            v = y;
        }
        int16_t in = (int16_t)(v * 30000.0);
        velvet_reverb_push_sample(in);

        if (switching && (i % (long)(0.15 * FS_OUT)) == 0) {
            oct = (int8_t)((oct == 1) ? -1 : oct + 1);
            velvet_reverb_set_octave(oct);
        }

        if ((i % FRAMES_PER_BLOCK) == (FRAMES_PER_BLOCK - 1)) velvet_reverb_poll();
        int16_t l = velvet_reverb_out_left();
        (void)velvet_reverb_out_right();

        if (l == 32767 || l == -32768) r.sat++;
        if (!isfinite((float)l)) r.nonfinite++;
        if (i >= total - meas) out[i - (total - meas)] = (float)l;
    }

    r.f220 = band(out, meas, 175.0, 250.0);
    r.f440 = band(out, meas, 350.0, 500.0);
    r.f880 = band(out, meas, 700.0, 1000.0);
    free(out);
    return r;
}

static double db(double a, double ref) { return 20.0 * log10((a + 1e-9) / ref); }

int main(void)
{
    printf("=== reverb octave shift, 350-500 Hz noise in ===\n");
    printf("band levels in dB relative to the 350-500 Hz band with the octave off\n\n");

    /* Throwaway first run: the very first render after start-up comes out
     * ~15 dB quiet whatever the octave setting (velvet_reverb_init doesn't
     * reset everything a previous run leaves behind), which would skew the
     * reference. */
    (void)run(0, 8.0, 0);

    Res off  = run(0,  8.0, 0);
    Res up   = run(1,  8.0, 0);
    Res down = run(-1, 8.0, 0);
    double ref = off.f440;

    printf("  %-8s %9s %9s %9s %6s\n", "octave", "175-250", "350-500", "700-1k", "sat");
    printf("  %-8s %9.1f %9.1f %9.1f %6ld\n", "off",  db(off.f220, ref),  db(off.f440, ref),  db(off.f880, ref),  off.sat);
    printf("  %-8s %9.1f %9.1f %9.1f %6ld\n", "up",   db(up.f220, ref),   db(up.f440, ref),   db(up.f880, ref),   up.sat);
    printf("  %-8s %9.1f %9.1f %9.1f %6ld\n", "down", db(down.f220, ref), db(down.f440, ref), db(down.f880, ref), down.sat);

    Res sw = run(0, 6.0, 1);
    printf("\n  switching every 150 ms: saturated samples %ld, non-finite %ld\n", sw.sat, sw.nonfinite);

    /* Equal blend: the octave band and the source band both near -3 dB.
     * (Octave down also halves the reverb's output HPF, so its octave band
     * isn't cut by it.)
     * The octave bands aren't empty with the shift off (the source filter's
     * skirts reach them), so check that switching on lifts them well clear. */
    int ok = (db(up.f880, ref) > -8.0) && (db(down.f220, ref) > -6.0)
          && (db(up.f440, ref) > -6.0) && (db(down.f440, ref) > -6.0)
          && (db(up.f880, off.f880) > 15.0) && (db(down.f220, off.f220) > 15.0)
          && sw.nonfinite == 0;
    printf("\n  RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
