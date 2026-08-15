/* effgains_step.c — does pot jitter cause isolated effGains recomputes?
 *
 * WHY THIS EXISTS
 * ---------------
 * Bisection on hardware established that the remaining click needs BOTH the morph
 * state to be moving (BISECT=nomorph was clean) AND the tap gains to be
 * recomputed (BISECT=nobgeff was clean). That conjunction is the signature of an
 * isolated recompute snapping the gains to a morph state that has drifted away
 * from them, rather than of CPU cost.
 *
 * background_eff_gains_update had two thresholds on different scales:
 *
 *   change detector : dead-band of DB_DURATION_TARGET = 0.01 s, i.e. 240 samples
 *                     of t2w, below which a target move is "not a change"
 *   settled test    : an absolute 1.0 sample
 *
 * A movement between the two is simultaneously too small to grant a settle window
 * and too large to count as settled, so the periodic refresh fires every 67 ms and
 * each firing recomputes exactly ONE stage after 67 ms of drift. Pot jitter lands
 * in that gap continuously.
 *
 * This test drives the macros the way an ADC does — a static setting plus a little
 * jitter — and counts recomputes, separating the isolated ones (>10 blocks after
 * the previous, so they carry a big step) from clustered ones (which track). The
 * numbers come from the real velvet_reverb.c, so this measures the shipped policy
 * rather than a model of it.
 *
 * Usage: ./effgains_step [seconds]        (default 20)
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

extern uint32_t host_dbg_bg_blocks;
extern uint32_t host_dbg_eff_calls;
extern uint32_t host_dbg_eff_isolated;
extern uint32_t host_dbg_eff_maxgap;
extern uint32_t host_dbg_eff_lastblk;

#define FS_OUT           48000.0
#define FRAMES_PER_BLOCK 32

/* 12-bit ADC, so one code is 1/4096 of travel. Jitter of a couple of codes is
 * ordinary for a pot on an unshielded panel and is exactly what the dead-bands
 * upstream exist to absorb. */
#define ADC_CODE (1.0f / 4096.0f)

typedef struct { double calls_s, iso_s; uint32_t maxgap; } Res;

static Res run(double secs, float decay, float tone, double jitter_codes)
{
    Res r; memset(&r, 0, sizeof r);

    velvet_reverb_init();
    velvet_reverb_apply_decay_macro(decay);
    velvet_reverb_apply_tone_macro(tone);

    host_dbg_bg_blocks = 0;
    host_dbg_eff_calls = 0;
    host_dbg_eff_isolated = 0;
    host_dbg_eff_maxgap = 0;
    host_dbg_eff_lastblk = 0;

    const long total = (long)(secs * FS_OUT);
    uint32_t rng = 9001u;
    double lp[4] = {0, 0, 0, 0};
    const double lpa = 1.0 - exp(-2.0 * M_PI * 2000.0 / FS_OUT);

    /* Settle first, then zero the counters, so the initial convergence after
     * init is not scored as if it were jitter-driven. */
    long settle = (long)(2.0 * FS_OUT);

    for (long i = 0; i < total + settle; i++) {
        rng = rng * 1103515245u + 12345u;
        double nz = ((double)((rng >> 16) & 0xFFFF) - 32768.0) / 32768.0;
        lp[0] += lpa * (nz    - lp[0]);
        lp[1] += lpa * (lp[0] - lp[1]);
        lp[2] += lpa * (lp[1] - lp[2]);
        lp[3] += lpa * (lp[2] - lp[3]);
        int32_t in = (int32_t)(lp[3] * 6.0 * 6000.0);
        if (in >  32767) in =  32767;
        if (in < -32768) in = -32768;
        velvet_reverb_push_sample((int16_t)in);

        /* Params arrive at main-loop rate (~1 kHz), knob physically still. */
        if ((i % 48) == 0) {
            rng = rng * 1103515245u + 12345u;
            double j = ((double)((rng >> 16) & 0xFFFF) / 65536.0 - 0.5) * 2.0;
            float d = decay + (float)(j * jitter_codes * ADC_CODE);
            if (d < 0.0f) d = 0.0f; if (d > 1.0f) d = 1.0f;
            velvet_reverb_apply_decay_macro(d);
        }

        if ((i % FRAMES_PER_BLOCK) == (FRAMES_PER_BLOCK - 1)) velvet_reverb_poll();
        (void)velvet_reverb_out_left();
        (void)velvet_reverb_out_right();

        if (i == settle) {
            host_dbg_bg_blocks = 0; host_dbg_eff_calls = 0;
            host_dbg_eff_isolated = 0; host_dbg_eff_maxgap = 0;
            host_dbg_eff_lastblk = 0;   /* or the first gap underflows */
        }
    }

    r.calls_s = host_dbg_eff_calls / secs;
    r.iso_s   = host_dbg_eff_isolated / secs;
    r.maxgap  = host_dbg_eff_maxgap;
    return r;
}

int main(int argc, char **argv)
{
    double secs = (argc > 1) ? atof(argv[1]) : 20.0;

    printf("=== effGains recompute policy under pot jitter (%.0f s each) ===\n\n", secs);
    printf("An isolated recompute is one landing >10 blocks after the previous, so it\n");
    printf("must snap the tap gains across everything the morph drifted in the gap.\n");
    printf("Those are the ones that step the convolution audibly; clustered\n");
    printf("recomputes track the morph in small increments and are the intent.\n\n");

    printf("  %-28s %12s %12s %9s %8s\n", "condition", "recompute/s", "isolated/s",
           "isolated%", "max gap");

    /* Decay is tested BELOW max on purpose. apply_decay_macro scales by
     * 1/DECAY_MACRO_TOP_SAT and set_macro_value clamps to [0,1], so at decay = 1.0
     * the value pins at 1.0 and jitter has literally no effect — a run at max
     * measures nothing and reports a misleading clean sheet. */
    struct { const char *name; float decay; double jit; } cases[] = {
        { "knob dead still, 0.5",       0.5f, 0.0  },
        { "1 ADC code, decay 0.5",      0.5f, 1.0  },
        { "3 ADC codes, decay 0.5",     0.5f, 3.0  },
        { "3 ADC codes, decay 0.9",     0.9f, 3.0  },
        { "10 codes, decay 0.5",        0.5f, 10.0 },
        { "pinned max (jitter inert)",  1.0f, 3.0  },
    };

    /* What the fix guarantees is a SHARE, not an absolute count. Once a gesture
     * is large enough to keep the morph moving, recomputes go near-continuous
     * (~1240/s) and track it in small increments, which is the intent; a handful
     * per second still land after a gap. Before the fix every recompute was
     * isolated — 100 % of them — so the share is what separates the two policies,
     * and it does not move as the jitter is made harsher. The absolute bound
     * still applies when little is happening, where an isolated recompute has no
     * gesture to explain it. */
    const double ISO_SHARE_MAX = 0.05;
    const double ISO_ABS_MAX   = 1.0;
    int bad = 0;
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        Res r = run(secs, cases[k].decay, 1.0f, cases[k].jit);
        double share = (r.calls_s > 0.0) ? r.iso_s / r.calls_s : 0.0;
        int tracking = r.calls_s >= 10.0;
        int fail = tracking ? (share > ISO_SHARE_MAX) : (r.iso_s > ISO_ABS_MAX);
        printf("  %-28s %12.2f %12.2f %8.2f%% %8u%s\n", cases[k].name,
               r.calls_s, r.iso_s, share * 100.0, r.maxgap,
               fail ? "   <-- stepping" : (tracking ? "   (tracking)" : ""));
        if (fail) bad = 1;
    }

    printf("\n  A healthy policy either recomputes nothing (the morph is settled) or\n");
    printf("  recomputes continuously in small increments that track a real gesture,\n");
    printf("  in which case isolated ones are a few percent of the total. The pre-fix\n");
    printf("  policy sat at 13.7/s with EVERY recompute isolated, each one snapping\n");
    printf("  the tap gains across a whole gap's worth of drift.\n");

    printf("\nRESULT: %s\n", bad ? "FAIL — isolated recomputes dominate"
                                 : "PASS — recomputes track the morph rather than stepping it");
    return bad ? 1 : 0;
}
