/* oct_render.c — render the reverb with its octave shift off, up and down,
 * for listening.
 *
 *   ./oct_render <prefix>     writes <prefix>_off.wav, <prefix>_up.wav,
 *                             <prefix>_down.wav (stereo, 48 kHz, reverb only)
 *
 * The source is synthesised so the renders are repeatable: a plucked-saw
 * arpeggio with a few percussive hits for ~9 s, then silence for the tail.
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

#define FS               48000
#define FRAMES_PER_BLOCK 32
#define SECS             14

static double source(long i)
{
    static const double notes[] = { 220.0, 277.18, 329.63, 440.0, 329.63, 277.18 };
    const long step = FS / 4;                    /* 4 notes a second */
    double t = (double)(i % step) / FS;
    long k = i / step;
    if (i >= (long)(9.0 * FS)) return 0.0;

    /* plucked saw: bright attack decaying into a softer tail */
    double f = notes[k % 6];
    double ph = fmod(t * f, 1.0);
    double saw = 2.0 * ph - 1.0;
    double s = saw * exp(-t * 6.0) * 0.5;

    /* a short noise hit every 2 s */
    if ((k % 8) == 0) {
        static uint32_t rng = 12345u;
        rng = rng * 1103515245u + 12345u;
        double nz = ((double)((rng >> 16) & 0xFFFF) - 32768.0) / 32768.0;
        s += nz * exp(-t * 40.0) * 0.6;
    }
    return s;
}

static void put16(FILE *f, uint16_t v) { fputc(v & 0xFF, f); fputc(v >> 8, f); }
static void put32(FILE *f, uint32_t v) { put16(f, v & 0xFFFF); put16(f, v >> 16); }

static void render(const char *path, int8_t octave)
{
    velvet_reverb_init();
    velvet_reverb_apply_decay_macro(0.6f);
    velvet_reverb_apply_tone_macro(0.7f);
    velvet_reverb_set_octave(octave);

    const long n = (long)SECS * FS;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite("RIFF", 1, 4, f); put32(f, 36 + (uint32_t)n * 4); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); put32(f, 16); put16(f, 1); put16(f, 2);
    put32(f, FS); put32(f, FS * 4); put16(f, 4); put16(f, 16);
    fwrite("data", 1, 4, f); put32(f, (uint32_t)n * 4);

    for (long i = 0; i < n; i++) {
        int32_t in = (int32_t)(source(i) * 16000.0);
        if (in > 32767) in = 32767;
        if (in < -32768) in = -32768;
        velvet_reverb_push_sample((int16_t)in);
        if ((i % FRAMES_PER_BLOCK) == (FRAMES_PER_BLOCK - 1)) velvet_reverb_poll();
        put16(f, (uint16_t)velvet_reverb_out_left());
        put16(f, (uint16_t)velvet_reverb_out_right());
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    const char *prefix = (argc > 1) ? argv[1] : "oct";
    char path[1024];

    /* Throwaway render: the first one after start-up comes out quiet (see
     * oct_pitch.c). */
    snprintf(path, sizeof path, "%s_warmup.wav", prefix);
    render(path, 0);
    remove(path);

    snprintf(path, sizeof path, "%s_off.wav", prefix);  render(path, 0);
    snprintf(path, sizeof path, "%s_up.wav", prefix);   render(path, 1);
    snprintf(path, sizeof path, "%s_down.wav", prefix); render(path, -1);
    printf("wrote %s_{off,up,down}.wav\n", prefix);
    return 0;
}
