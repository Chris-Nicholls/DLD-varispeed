#!/usr/bin/env python3
"""
diag_decode.py — FSK demodulator + event-stream parser for the SWN
firmware diagnostic firehose.

The firmware modulates the right audio channel with an FSK bit stream
(see inc/diag_fsk.h).  This script reads any common WAV format,
extracts zero-crossings, classifies each run as a '0', '1', or pause
symbol, reassembles bits into bytes, finds packet sync, verifies the
CRC-16-CCITT, and decodes the payload into (event_type, cycle_count)
tuples.

Robust against:
  - AC coupling (we track and subtract a slow moving baseline).
  - Amplitude attenuation / level normalisation (only zero-crossings
    matter).
  - 16-bit truncation (zero-crossings survive arbitrary bit-depth
    reduction).
  - Sample-rate variations (run-length thresholds scale with the
    capture's reported sample rate).

Wire format:
    Symbol  Period @ 48 kHz
    ------  ---------------
    '0'     4 samples then flip
    '1'     8 samples then flip
    pause   16 samples then flip   (idle marker, carries no bit)

Packet:
    [0xA5][0x5A][len][payload]  [crc16_hi][crc16_lo]
                                ^ CRC16-CCITT(0xFFFF) over [len][payload]

Event (3 bytes, big-endian):
    bits 23-20 : evt_type  (4 bits → 16 slots; see EVT_NAMES below)
    bits 19- 0 : cycles    (saturates at 0xFFFFF ≈ 6.2 ms @ 168 MHz)

Usage:
    python3 diag_decode.py capture.wav [--plot] [--save-plots DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from dataclasses import dataclass

import numpy as np

CPU_HZ = 168_000_000  # DLD STM32F427

EVT_NAMES = {
    0:  "ch0_ISR_(L)",
    1:  "ch1_ISR_(R)",
    2:  "reverb_block",     # full main-loop reverb block process
    3:  "reverb_drop",      # input-block drops
    4:  "reverb_T0",        # do_t0_phase (early reflections + per-tap LFO)
    5:  "reverb_T2",        # do_t2_phase (late, stereo)
    6:  "reverb_T1",        # do_t1_phase (middle)
    7:  "reverb_morph",     # update_morph_state
    8:  "reverb_finalize",  # do_finalize (HPF/LPF + upsample + mix)
    9:  "reverb_bg_eff",    # background_eff_gains_update (post-block, one stage per call)
    10: "reverb_predelay",  # do_predelay (2-line feedback sustain engine)
    11: "isr_in_block",      # codec-ISR cycles preempting one reverb block (block - this = pure compute)
    12: "isr_sdram_read",   # time in memory_read[_varispeed] per ISR call (prefetchable)
    13: "isr_sdram_write",  # time in memory_write[_fade] per ISR call
    14: "output_miss",      # cumulative bitcrush miss count (emitted ~1/s; max = total)
    15: "poll_latency",     # block_ready -> poll pickup lag (wall cycles)
}

# Firmware built with -DDIAG_REVERB_PROFILE repurposes the two delay-engine
# SDRAM slots (the 4-bit wire field has no spare ones) to break the T2 stage
# down. Pass --profile to label them correctly.
EVT_NAMES_PROFILE = {
    **EVT_NAMES,
    12: "t2_dma_wait",      # cycles/block spun in dma2_wait (summed over 64 taps)
    13: "t2_dma_kick",      # cycles/block in dma2_kick register setup (64 taps)
}

# Wire constants (mirrors inc/diag_fsk.h / diag_log.h).
ZERO_PERIOD_REF = 4
ONE_PERIOD_REF = 8
PAUSE_PERIOD_REF = 16
REF_FS = 48_000.0

SYNC0 = 0xA5
SYNC1 = 0x5A
EVT_BITS = 4
CYCLES_BITS = 20
CYCLES_MASK = (1 << CYCLES_BITS) - 1
EVT_MASK = (1 << EVT_BITS) - 1


# ───────────────────────── WAV reading ──────────────────────────────


def read_right_channel(path: str) -> tuple[np.ndarray, int]:
    """Load the right channel of a WAV file as float64 in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
        scale = 1.0 / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        scale = 1.0 / (1 << 15)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        a = (b[:, 0].astype(np.int32)
             | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        a = np.where(a & 0x800000, a | ~0xFFFFFF, a)
        scale = 1.0 / (1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        scale = 1.0 / (1 << 31)
    else:
        raise ValueError(f"Unsupported sample width {sw}")

    a = a.reshape(-1, n_ch)
    right = a[:, n_ch - 1] if n_ch > 1 else a[:, 0]
    return right.astype(np.float64) * scale, sr


# ───────────────────────── FSK demodulation ─────────────────────────


def remove_baseline(sig: np.ndarray, win: int) -> np.ndarray:
    """Subtract a centred moving-mean baseline.  AC coupling and
    long capture sessions can introduce slow DC drift; this kills it
    without disturbing the FSK band (lowest FSK fundamental is the
    pause rate ≈ 1.5 kHz at 48 kHz, while typical audio AC-coupling
    rolloff is at a few Hz)."""
    if win <= 1 or win >= len(sig):
        return sig - np.mean(sig)
    pad = win // 2
    cs = np.cumsum(np.concatenate(([0.0], sig)))
    avg = (cs[win:] - cs[:-win]) / win
    head = np.full(pad, avg[0])
    tail = np.full(len(sig) - len(avg) - pad, avg[-1])
    base = np.concatenate([head, avg, tail])
    return sig - base


def detect_runs(sig: np.ndarray, hyst: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect alternating-polarity runs with hysteresis.

    Returns (run_lengths, run_polarities) — polarity is +1 / -1.
    Hysteresis prevents tiny noise wiggles from creating spurious
    crossings near zero.  Set `hyst` ~10 % of the post-baseline
    signal RMS.
    """
    state = np.zeros(len(sig), dtype=np.int8)
    cur = 0  # 0 = unknown, +1 / -1 once we cross either threshold
    for i in range(len(sig)):
        v = sig[i]
        if cur >= 0 and v < -hyst:
            cur = -1
        elif cur <= 0 and v > hyst:
            cur = 1
        state[i] = cur
    # Find run boundaries.
    nonzero = state != 0
    if not nonzero.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int8)
    first = int(np.argmax(nonzero))
    s = state[first:]
    diff = np.diff(s)
    boundaries = np.nonzero(diff != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(s)]))
    lens = (ends - starts).astype(np.int32)
    polarities = s[starts].astype(np.int8)
    return lens, polarities


def classify_runs(run_lengths: np.ndarray, sr: int) -> np.ndarray:
    """Classify each run as a symbol: 0='0', 1='1', 2=pause.

    Thresholds scale with the capture's sample rate so a 44.1 kHz
    recording decodes too.  The reference periods at 48 kHz are 4/8/16
    samples; we set midpoint thresholds between them.
    """
    scale = sr / REF_FS
    z = ZERO_PERIOD_REF * scale
    o = ONE_PERIOD_REF * scale
    p = PAUSE_PERIOD_REF * scale
    thr_01 = (z + o) / 2.0
    thr_1p = (o + p) / 2.0
    syms = np.where(
        run_lengths < thr_01, 0,
        np.where(run_lengths < thr_1p, 1, 2),
    ).astype(np.int8)
    return syms


def symbols_to_bits(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the symbol stream onto a (bit, source_run_idx) sequence.
    Pause symbols are dropped (they don't encode bits).  source_run_idx
    gives the run index in `symbols` that produced each bit, so we can
    jump back later if we want to anchor packets to time."""
    mask = symbols != 2
    bits = symbols[mask].astype(np.int8)
    src = np.nonzero(mask)[0]
    return bits, src


# ───────────────────────── packet parsing ───────────────────────────


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def bits_to_byte(bits: np.ndarray) -> int:
    """Convert 8 bits (MSB-first) to an unsigned byte."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


@dataclass
class ParseStats:
    n_packets_ok: int = 0
    n_packets_bad_crc: int = 0
    n_resyncs: int = 0
    n_truncated: int = 0


def find_packets(bits: np.ndarray, src: np.ndarray, max_payload: int = 24
                 ) -> tuple[list[tuple[int, int, int]], ParseStats]:
    """Scan the bit stream for sync patterns; on each, attempt to
    decode a packet.  Returns:
       events  = list of (evt_type, cycles, run_idx_of_packet_start)
       stats   = ParseStats
    """
    events: list[tuple[int, int, int]] = []
    stats = ParseStats()

    n = len(bits)
    if n < 16:
        return events, stats

    # Pre-bake the 16-bit sync pattern as a uint16 we'll match
    # bit-by-bit using a sliding bitwise shift register.
    SYNC = (SYNC0 << 8) | SYNC1
    reg = 0
    mask16 = 0xFFFF

    i = 0
    while i < n:
        # Refill the shift register up to 16 bits ahead of i.
        # (We handle search as: examine bits[i] and prior 15 bits.)
        if i < 16:
            for j in range(i + 1):
                reg = ((reg << 1) | int(bits[j])) & mask16
        else:
            reg = ((reg << 1) | int(bits[i])) & mask16

        if reg == SYNC and i + 1 + 8 <= n:
            # Bit i is the last bit of SYNC1.  The next bit starts
            # the length byte.
            cur = i + 1
            if cur + 8 > n:
                stats.n_truncated += 1
                break
            length = bits_to_byte(bits[cur:cur + 8]); cur += 8
            if length == 0 or length > max_payload or (length % 3) != 0:
                # Implausible length — keep scanning bit-by-bit.
                i += 1
                continue
            if cur + 8 * length + 16 > n:
                stats.n_truncated += 1
                break
            payload_bits = bits[cur:cur + 8 * length]; cur += 8 * length
            crc_bits = bits[cur:cur + 16]; cur += 16

            # Reassemble payload bytes.
            payload = bytes(
                bits_to_byte(payload_bits[k:k + 8])
                for k in range(0, 8 * length, 8)
            )
            crc_rx = (bits_to_byte(crc_bits[:8]) << 8) | bits_to_byte(crc_bits[8:])
            crc_calc = crc16_ccitt(bytes([length]) + payload)
            if crc_rx != crc_calc:
                stats.n_packets_bad_crc += 1
                stats.n_resyncs += 1
                i += 1  # bit-slip past this false sync
                continue

            stats.n_packets_ok += 1
            anchor_run_idx = int(src[i]) if i < len(src) else -1
            for k in range(0, length, 3):
                b0, b1, b2 = payload[k], payload[k + 1], payload[k + 2]
                packed = (b0 << 16) | (b1 << 8) | b2
                evt = (packed >> CYCLES_BITS) & EVT_MASK
                cyc = packed & CYCLES_MASK
                events.append((evt, cyc, anchor_run_idx))
            i = cur  # advance past the consumed packet
            continue

        i += 1

    return events, stats


# ───────────────────────── pipeline ─────────────────────────────────


@dataclass
class DecodeResult:
    sample_rate: int
    duration_s: float
    runs_total: int
    pause_runs: int
    events: list[tuple[int, int, int]]
    parse_stats: ParseStats


def decode_wav(path: str, hyst_frac: float = 0.10,
               baseline_ms: float = 5.0) -> DecodeResult:
    sig, sr = read_right_channel(path)
    duration = len(sig) / sr

    win = max(1, int(baseline_ms * 1e-3 * sr))
    sig_dc = remove_baseline(sig, win)

    # Hysteresis from RMS so we work for any amplitude.
    rms = float(np.sqrt(np.mean(sig_dc ** 2)) + 1e-12)
    hyst = rms * hyst_frac

    run_lens, _pol = detect_runs(sig_dc, hyst)
    syms = classify_runs(run_lens, sr)
    bits, src = symbols_to_bits(syms)
    events, stats = find_packets(bits, src)

    return DecodeResult(
        sample_rate=sr,
        duration_s=duration,
        runs_total=len(run_lens),
        pause_runs=int(np.sum(syms == 2)),
        events=events,
        parse_stats=stats,
    )


def cycles_to_us(c) -> float:
    return float(c) * 1e6 / CPU_HZ


# ───────────────────────── reporting ────────────────────────────────


def print_report(res: DecodeResult, names: dict[int, str] = EVT_NAMES) -> None:
    print(f"capture: {res.duration_s:.2f} s @ {res.sample_rate} Hz")
    print(f"  runs detected   : {res.runs_total}")
    print(f"  pause runs      : {res.pause_runs}")
    print(f"  packets OK      : {res.parse_stats.n_packets_ok}")
    print(f"  packets bad-CRC : {res.parse_stats.n_packets_bad_crc}")
    print(f"  truncated       : {res.parse_stats.n_truncated}")
    print(f"  resyncs         : {res.parse_stats.n_resyncs}")
    print(f"  events decoded  : {len(res.events)}")
    print()

    if not res.events:
        print("(no events decoded — check that the right audio channel was "
              "captured and that the firmware was built with diag_log_enabled=1)")
        return

    by_type: dict[int, list[int]] = {k: [] for k in names}
    for evt, cyc, _ in res.events:
        by_type.setdefault(evt, []).append(cyc)

    header = (
        f"{'event':<18}{'count':>8}{'rate/s':>10}"
        f"{'min µs':>10}{'mean µs':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'max µs':>10}"
    )
    print(header)
    print("-" * len(header))
    for eid, name in names.items():
        vals = np.array(by_type.get(eid, []), dtype=np.int64)
        n = len(vals)
        if n == 0:
            print(f"{name:<18}{0:>8}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")
            continue
        us = vals * 1e6 / CPU_HZ
        rate = n / max(res.duration_s, 1e-9)
        print(
            f"{name:<18}{n:>8}{rate:>10.1f}"
            f"{us.min():>10.1f}{us.mean():>10.1f}"
            f"{np.percentile(us, 50):>10.1f}"
            f"{np.percentile(us, 95):>10.1f}"
            f"{np.percentile(us, 99):>10.1f}"
            f"{us.max():>10.1f}"
        )


def print_profile_summary(res: DecodeResult) -> None:
    """Derived figures for a -DDIAG_REVERB_PROFILE capture.

    Because profile builds sample whole blocks, every stage below comes from
    the same population of blocks and can legitimately be summed and compared
    against the block total.
    """
    by_type: dict[int, list[int]] = {}
    for evt, cyc, _ in res.events:
        by_type.setdefault(evt, []).append(cyc)

    def stat(eid: int, fn=np.mean) -> float | None:
        v = by_type.get(eid)
        if not v:
            return None
        return float(fn(v)) * 1e6 / CPU_HZ

    def fmt(x: float | None, unit: str = " µs") -> str:
        return "—" if x is None else f"{x:.1f}{unit}"

    block = stat(2)
    isr_in = stat(11)
    t2 = stat(5)
    wait = stat(12)
    kick = stat(13)

    print()
    print("=" * 62)
    print("PROFILE SUMMARY")
    print("=" * 62)

    if block is None:
        print("no reverb_block events — was this built with PROFILE=1?")
        return

    compute = block - isr_in if isr_in is not None else None
    print(f"block deadline        : {BLOCK_BUDGET_US:.1f} µs")
    print(f"block wall-clock mean : {fmt(block)}  ({100*block/BLOCK_BUDGET_US:.1f}% of deadline)")
    if isr_in is not None:
        print(f"  codec ISR preemption: {fmt(isr_in)}")
        print(f"  pure reverb compute : {fmt(compute)}"
              f"  -> {100*compute/BLOCK_BUDGET_US:.1f}% of one core")
    p95 = stat(2, lambda v: np.percentile(v, 95))
    p99 = stat(2, lambda v: np.percentile(v, 99))
    mx = stat(2, np.max)
    print(f"block p95/p99/max     : {fmt(p95)} / {fmt(p99)} / {fmt(mx)}")

    print()
    print("stage breakdown (ISR time already subtracted):")
    accounted = 0.0
    for eid, label in ((7, "morph"), (10, "predelay"), (4, "T0"),
                       (6, "T1"), (5, "T2 (incl. sat)"), (8, "finalize")):
        v = stat(eid)
        if v is None:
            print(f"  {label:<16}      —")
            continue
        accounted += v
        pct = 100 * v / compute if compute else 0.0
        print(f"  {label:<16}{v:>8.1f} µs  ({pct:4.1f}% of compute)")
    if compute is not None:
        rest = compute - accounted
        print(f"  {'unaccounted':<16}{rest:>8.1f} µs  "
              f"({100*rest/compute:4.1f}%)  <- bridges, input write, log overhead")

    print()
    print("T2 decomposition — the question this build exists to answer:")
    if t2 is None or wait is None or kick is None:
        print("  incomplete (need reverb_T2 + t2_dma_wait + t2_dma_kick)")
    else:
        mac = t2 - wait - kick
        print(f"  dma2_wait  (spin)  {wait:>8.1f} µs  ({100*wait/t2:4.1f}% of T2)")
        print(f"  dma2_kick  (setup) {kick:>8.1f} µs  ({100*kick/t2:4.1f}% of T2)")
        print(f"  convolution + sat  {mac:>8.1f} µs  ({100*mac/t2:4.1f}% of T2)")
        print()
        if wait > 0.25 * t2:
            print("  => TRANSFER-BOUND. The prefetch is slower than the arithmetic it")
            print("     overlaps, so double-buffering cannot hide it. Widening the")
            print("     stream to 32-bit + INCR4 bursts should collapse this.")
        elif wait < 0.05 * t2:
            print("  => COMPUTE-BOUND. The prefetch is fully hidden; the DMA is doing")
            print("     its job. Optimise the convolution inner loop instead.")
        else:
            print("  => Marginal. Prefetch is mostly hidden; modest gains available.")

    drops = len(by_type.get(3, []))
    misses = by_type.get(14, [])
    print()
    print(f"block drops           : {drops}")
    print(f"output miss (cumul.)  : {max(misses) if misses else 0}")


def try_plot(res: DecodeResult, outdir: str | None) -> None:
    try:
        import matplotlib
        if outdir is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
        return

    by_type: dict[int, tuple[list[int], list[int]]] = {k: ([], []) for k in EVT_NAMES}
    # Approx event time = run-anchor index / sample_rate; we don't have
    # per-bit timestamps, but the run start index gives us close enough.
    # Without it we'd have to reconstruct timing from run lengths.
    for evt, cyc, anchor in res.events:
        cycs, anchors = by_type[evt]
        cycs.append(cyc)
        anchors.append(anchor)

    fig, axes = plt.subplots(len(EVT_NAMES), 2, figsize=(12, 10), tight_layout=True)
    for row, (eid, name) in enumerate(EVT_NAMES.items()):
        cycs, anchors = by_type[eid]
        ax_t = axes[row, 0]
        ax_h = axes[row, 1]
        if cycs:
            us = np.array(cycs) * 1e6 / CPU_HZ
            ax_t.plot(np.arange(len(us)), us, ".", markersize=2)
            ax_h.hist(us, bins=80, log=True)
        ax_t.set_title(f"{name} — sequence (n={len(cycs)})")
        ax_t.set_xlabel("event #")
        ax_t.set_ylabel("µs")
        ax_t.grid(True, alpha=0.3)
        ax_h.set_title(f"{name} — histogram")
        ax_h.set_xlabel("µs")
        ax_h.set_ylabel("count (log)")
        ax_h.grid(True, alpha=0.3)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "diag_decode.png")
        fig.savefig(path, dpi=110)
        print(f"\nsaved plot → {path}")
    else:
        plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wav", help="WAV capture (right channel = FSK stream)")
    ap.add_argument("--plot", action="store_true", help="show matplotlib plots")
    ap.add_argument("--save-plots", metavar="DIR",
                    help="save plots to DIR (non-interactive backend)")
    ap.add_argument("--hyst", type=float, default=0.10,
                    help="zero-crossing hysteresis as a fraction of RMS (default: 0.10)")
    ap.add_argument("--baseline-ms", type=float, default=5.0,
                    help="moving-mean window for baseline removal in ms (default: 5)")
    ap.add_argument("--profile", action="store_true",
                    help="capture came from a PROFILE=1 build: label the repurposed "
                         "slots 12/13 as the T2 DMA breakdown and print the summary")
    args = ap.parse_args()

    res = decode_wav(args.wav, hyst_frac=args.hyst, baseline_ms=args.baseline_ms)
    print_report(res, EVT_NAMES_PROFILE if args.profile else EVT_NAMES)
    if args.profile:
        print_profile_summary(res)
    if args.plot or args.save_plots:
        try_plot(res, args.save_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
