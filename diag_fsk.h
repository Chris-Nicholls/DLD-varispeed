/*
 * diag_fsk.h — FSK transmitter for DLD diagnostic firehose
 *
 * Ported from SWN; amplitudes scaled for DLD's 16-bit codec output.
 *
 * Symbol → on-wire shape (state alternates ±AMP between symbols):
 *     '0'   : hold for ZERO_PERIOD samples, then flip
 *     '1'   : hold for ONE_PERIOD  samples, then flip
 *     'P'   : hold for PAUSE_PERIOD samples, then flip (idle marker)
 *
 * Packet:
 *     [0xA5][0x5A][len][payload][crc16_hi][crc16_lo]
 * Each event = 24 bits = [evt:2][cycles:22].
 *
 * Drives the right channel one sample per audio frame.
 */

#pragma once

#include <stdint.h>

#define DIAG_FSK_ZERO_PERIOD  4u
#define DIAG_FSK_ONE_PERIOD   8u
#define DIAG_FSK_PAUSE_PERIOD 16u

/* ±0x6000 ≈ -3 dBFS at 16-bit signed. */
#define DIAG_FSK_AMP_POS  ( 0x6000)
#define DIAG_FSK_AMP_NEG  (-0x6000)

#define DIAG_FSK_SYNC0    0xA5u
#define DIAG_FSK_SYNC1    0x5Au

#define DIAG_FSK_MAX_EVENTS_PER_PACKET 4u

#define DIAG_FSK_PACKET_MAX_BYTES \
    (3u + (3u * DIAG_FSK_MAX_EVENTS_PER_PACKET) + 2u)

void   diag_fsk_init(void);
int16_t diag_fsk_next_sample(void);
