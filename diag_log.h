/*
 * diag_log.h — firehose event log for DLD reverb instrumentation
 *
 * Ported from SWN.  Producers (any ISR) call diag_log(evt, cycles).
 * A separate FSK transmitter drains the ring buffer and shoves the
 * events out a send-jack as a Manchester/FSK bit stream that survives
 * any normal audio recording chain.  Capture the jack at 48 kHz,
 * decode with test/diag_decode.py.
 *
 * On-wire event format (3 bytes, big-endian):
 *     byte 0 : [evt_type:4][cycles_hi:4]
 *     byte 1 : [cycles_mid:8]
 *     byte 2 : [cycles_lo:8]
 *   → 20-bit cycles, saturates at 0xFFFFF (~5.8 ms @ 180 MHz).
 *     16 event slots — keep `test/diag_decode.py` in sync.
 */

#pragma once

#include <stdint.h>
#include "stm32f4xx.h"

typedef enum {
    DIAG_EVT_AUDIOISR_CH0    =  0u, /* process_audio_block_codec channel 0     */
    DIAG_EVT_AUDIOISR_CH1    =  1u, /* process_audio_block_codec channel 1     */
    DIAG_EVT_REVERB_BLOCK    =  2u, /* main-loop reverb block processing       */
    DIAG_EVT_REVERB_DROP     =  3u, /* main loop missed deadline (cycles = 0)  */
    DIAG_EVT_REVERB_T0       =  4u, /* do_t0_phase (early refl. + per-tap LFO)  */
    DIAG_EVT_REVERB_T2       =  5u, /* do_t2_phase (late, stereo)              */
    DIAG_EVT_REVERB_T1       =  6u, /* do_t1_phase (middle)                    */
    DIAG_EVT_REVERB_MORPH    =  7u, /* update_morph_state cycles               */
    DIAG_EVT_REVERB_FINALIZE =  8u, /* do_finalize (HPF/LPF + upsample + mix)  */
    DIAG_EVT_REVERB_BG_EFF   =  9u, /* background_eff_gains_update (post-block, one stage per call) */
    DIAG_EVT_REVERB_PREDELAY = 10u, /* do_predelay (2-line feedback sustain engine) */
    DIAG_EVT_REVERB_ISR_IN_BLOCK = 11u, /* codec-ISR cycles that preempted one reverb block */
    DIAG_EVT_ISR_SDRAM_READ  = 12u, /* time in memory_read[_varispeed] (SDRAM delay-line reads) */
    DIAG_EVT_ISR_SDRAM_WRITE = 13u, /* time in memory_write[_fade] (SDRAM delay-line writes) */
    DIAG_EVT_OUTPUT_MISS     = 14u, /* cumulative count: output read repeated a stale block (bitcrush) */
    DIAG_EVT_POLL_LATENCY    = 15u, /* cycles from block_ready set -> poll pickup (main-loop lag) */
    DIAG_EVT_COUNT           = 16u,
} diag_evt_t;

#define DIAG_CYCLES_BITS   20u
#define DIAG_CYCLES_MASK   ((1u << DIAG_CYCLES_BITS) - 1u)
#define DIAG_EVT_SHIFT     DIAG_CYCLES_BITS
#define DIAG_EVT_MASK      0xFu

#define DIAG_LOG_SIZE 2048u
#define DIAG_LOG_MASK (DIAG_LOG_SIZE - 1u)

extern volatile uint32_t diag_log_buf[DIAG_LOG_SIZE];
extern volatile uint32_t diag_log_head;
extern volatile uint32_t diag_log_tail;
extern volatile uint32_t diag_log_dropped;

extern volatile uint32_t diag_thresh_cycles[DIAG_EVT_COUNT];
extern volatile uint32_t diag_evt_total[DIAG_EVT_COUNT];
extern volatile uint8_t  diag_log_enabled;

/* Running total of cycles spent inside the audio ISRs (process_audio_block_codec).
 * Each ISR adds its own measured duration here. The main-loop reverb stage
 * timers sample this before/after a stage and subtract the delta, so per-stage
 * cycle counts report *pure compute* rather than wall-clock-including-ISR
 * (the reverb runs in the main loop and is preempted by the codec ISRs). */
extern volatile uint32_t diag_isr_cycles;

/* Producer.  Safe from any ISR priority. */
static inline void diag_log(diag_evt_t evt, uint32_t cycles)
{
    diag_evt_total[evt]++;
    if (!diag_log_enabled) return;
    if (cycles < diag_thresh_cycles[evt]) return;

    if (cycles > DIAG_CYCLES_MASK) cycles = DIAG_CYCLES_MASK;
    uint32_t packed = ((uint32_t)evt << DIAG_EVT_SHIFT) | cycles;

    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint32_t h = diag_log_head;
    uint32_t next = (h + 1u) & DIAG_LOG_MASK;
    if (next != diag_log_tail) {
        diag_log_buf[h] = packed;
        diag_log_head = next;
    } else {
        diag_log_dropped++;
    }
    if (!primask) __enable_irq();
}

/* Consumer: single FSK TX caller. */
static inline int diag_log_pop_packed(uint32_t *out)
{
    uint32_t t = diag_log_tail;
    if (t == diag_log_head) return 0;
    *out = diag_log_buf[t] & 0xFFFFFFu;
    diag_log_tail = (t + 1u) & DIAG_LOG_MASK;
    return 1;
}
