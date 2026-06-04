/*
 * diag_log.c — backing storage + threshold defaults.
 *
 * Defaults chosen so the FSK channel (~250 events/s bandwidth) stays
 * within budget while still capturing every outlier.  Override at run
 * time by writing to diag_thresh_cycles[] from main() or a debugger.
 *
 *   AUDIOISR_CH0 > 50 µs (9000 cycles @ 180 MHz)  — flag any heavy ISR
 *   AUDIOISR_CH1 > 50 µs
 *   PHASE_T1     > 20 µs (3600 cycles)
 *   PHASE_T2     > 20 µs
 */

#include "diag_log.h"

volatile uint32_t diag_log_buf[DIAG_LOG_SIZE];
volatile uint32_t diag_log_head    = 0;
volatile uint32_t diag_log_tail    = 0;
volatile uint32_t diag_log_dropped = 0;
volatile uint32_t diag_evt_total[DIAG_EVT_COUNT] = { 0 };

volatile uint32_t diag_thresh_cycles[DIAG_EVT_COUNT] = {
    /* Outlier capture: only log events that exceed the budget so the FSK
     * channel stays inside its ~300 ev/s envelope. 168 MHz CPU; 1 µs ≈ 168
     * cycles. Tighten/loosen by writing this array at runtime if needed. */
    [DIAG_EVT_AUDIOISR_CH0]     = 9000u,   /* > ~54 µs */
    [DIAG_EVT_AUDIOISR_CH1]     = 9000u,
    [DIAG_EVT_REVERB_BLOCK]     = 84000u,  /* > 500 µs (budget is 666 µs)  */
    [DIAG_EVT_REVERB_DROP]      = 0u,      /* always log block drops       */
    [DIAG_EVT_REVERB_T0]        = 3600u,   /* > ~21 µs */
    [DIAG_EVT_REVERB_T1]        = 3600u,
    [DIAG_EVT_REVERB_T2]        = 3600u,
    [DIAG_EVT_REVERB_MORPH]     = 3600u,
    [DIAG_EVT_REVERB_FINALIZE]  = 3600u,
    [DIAG_EVT_REVERB_BG_EFF]    = 3600u,
    [DIAG_EVT_REVERB_T0_RECIRC] = 3600u,
    [DIAG_EVT_REVERB_T1_RECIRC] = 3600u,
    [DIAG_EVT_REVERB_T2_RECIRC] = 3600u,
    [DIAG_EVT_RESERVED_13]      = 0xFFFFFu,
    [DIAG_EVT_RESERVED_14]      = 0xFFFFFu,
    [DIAG_EVT_RESERVED_15]      = 0xFFFFFu,
};

volatile uint8_t diag_log_enabled = 1;
