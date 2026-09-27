/*
 * looping_delay.c - the heart of the DLD
 * Functions for processing audio buffer from the codec, managing audio buffer addresses,
 * cross-fades, windowing/scrolling, and reverse
 *
 * Author: Dan Green (danngreen1@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * See http://creativecommons.org/licenses/MIT/ for more information.
 *
 * -----------------------------------------------------------------------------
 */

#include <string.h>
#include "globals.h"
#include "looping_delay.h"
#include "sdram.h"
#include "velvet_reverb.h"
#ifdef DIAG_FSK_ENABLE
#include "diag_fsk.h"
#endif

/* Per-channel effective loop sizes — see looping_delay.h */
uint32_t loop_size[NUM_CHAN];
#include "adc.h"
#include "params.h"
#include "audio_memory.h"
#include "timekeeper.h"
#include "compressor.h"
#include "leds.h"
#include "dig_pins.h"
#include "codec_CS4271.h"

extern const float epp_lut[4096];
extern float param[NUM_CHAN][NUM_PARAMS];
extern uint8_t mode[NUM_CHAN][NUM_CHAN_MODES];
extern uint8_t global_mode[NUM_GLOBAL_MODES];
extern float global_param[NUM_GLOBAL_PARAMS];


extern uint8_t flag_inf_change[2];

uint8_t SAMPLESIZE=2;

extern uint8_t flag_pot_changed_revdown[NUM_POT_ADCS];

extern int16_t CODEC_DAC_CALIBRATION_DCOFFSET[4];
//extern int16_t CODEC_ADC_CALIBRATION_DCOFFSET[4];

volatile uint32_t ping_time;
uint32_t locked_ping_time[NUM_CHAN];
volatile uint32_t divmult_time[NUM_CHAN];

uint32_t write_addr[NUM_CHAN];
uint32_t read_addr[NUM_CHAN];

uint32_t loop_start[NUM_CHAN];
uint32_t loop_end[NUM_CHAN];

const uint32_t LOOP_RAM_BASE[NUM_CHAN] = {SDRAM_BASE, SDRAM_BASE + LOOP_SIZE};


uint32_t fade_queued_dest_divmult_time[NUM_CHAN];
uint8_t queued_write_fade_state[NUM_CHAN];
uint32_t fade_queued_dest_read_addr[NUM_CHAN];
uint32_t fade_queued_dest_write_addr[NUM_CHAN];
uint32_t fade_dest_read_addr[NUM_CHAN];
uint32_t fade_dest_write_addr[NUM_CHAN];
float read_fade_pos[NUM_CHAN];
float write_fade_pos[NUM_CHAN];

uint8_t doing_reverse_fade[NUM_CHAN] = {0,0};

// Varispeed state
float fractional_read_pos[NUM_CHAN] = {0.0f, 0.0f};
float read_speed[NUM_CHAN] = {1.0f, 1.0f};
uint32_t target_read_addr[NUM_CHAN];

/* Varispeed glide state. See update_read_speed(). */
enum VarispeedStates {
	VS_SETTLED,		// at the target, or nudging towards it
	VS_WAITING,		// a big change is queued for the next clock tick
	VS_GLIDING,		// moving at exactly 2x or 0.5x
	VS_CROSSFADE	// glide used up its budget: crossfade the rest of the way
};
static uint8_t vs_state[NUM_CHAN] = {VS_SETTLED, VS_SETTLED};

/* How much more delay-time change, in samples, the current glide may cover
 * before handing over to a crossfade. */
static float glide_budget[NUM_CHAN] = {0.0f, 0.0f};

/* Delay-time errors up to this size are nudged out with an inaudible speed
 * change, not a glide. They are what clock jitter, and the small drift of a
 * slowly averaged ping, produce. A real TIME change is a jump between clock
 * divisions, which is always far bigger than this. */
#define VARISPEED_NUDGE_MAX_SAMPLES  240		/* 5 ms */
#define VARISPEED_NUDGE_MAX_FRAC     32		/* ...or 1/32 of the delay time, whichever is bigger */

/* Largest speed change a nudge uses: 0.3%, about 5 cents. */
#define VARISPEED_NUDGE_RATE         0.003f

/* Within this distance (in samples) of the target, the read head is placed
 * exactly on it. Must be more than the smallest nudge memory_read_varispeed
 * will actually move for: speeds within 0.001 of 1x take its 1x fast path,
 * which on a 16-sample block ignores anything under 0.016 samples. */
#define VARISPEED_ARRIVED            0.02f

/* A glide covers at most this many clock periods of delay-time change: 2
 * periods at 0.5x, or 1 at 2x. Anything left over is crossfaded. Without a
 * limit, a big jump such as the time switch's +16 would take 32 clock periods
 * at 0.5x, because lengthening a delay by N samples takes at least N samples
 * whatever the speed. */
#define VARISPEED_GLIDE_MAX_PERIODS  1

/* Send/Return insert in the delay loop.
 *
 * The SEND jack carries the delay buffer read. Right of centre on the MIX
 * pot, the RETURN jack stands in for that read: it feeds both the regen
 * path and the wet output. Left of centre, the return is bypassed.
 *
 * A trip out of SEND and back in RETURN costs SEND_RETURN_LATENCY_SAMPLES:
 * two DMA half-buffers (the block written now plays next block, and is
 * captured and handed to us the block after that) plus the CS4271 DAC and ADC
 * group delays. To keep repeat timing the same, the read head runs that
 * many samples closer to the write head. The left-of-centre path then puts
 * the read through an internal line of the same length, which stands in for
 * the cable. So the two routings stay time-aligned. With SEND patched to
 * RETURN at unity gain, both sides of the knob sound the same, and crossing
 * 12 o'clock never moves the read head.
 *
 * The block part is exact. The codec part is taken from the datasheet and
 * should be confirmed on hardware (patch SEND->RETURN, ping a short delay and
 * compare the repeat spacing on either side of centre). */
#define CODEC_DAC_GROUP_DELAY        9
#define CODEC_ADC_GROUP_DELAY        12
#define SEND_RETURN_LATENCY_SAMPLES  (2 * (codec_BUFF_LEN / 8) + CODEC_DAC_GROUP_DELAY + CODEC_ADC_GROUP_DELAY)

/* Below this delay time there is no room to read that far ahead, so
 * compensation is dropped. Both routings then run uncompensated: the left
 * path is exactly the old behaviour, and the right path adds the latency. */
#define SEND_RETURN_MIN_COMP_TIME    (SEND_RETURN_LATENCY_SAMPLES + codec_BUFF_LEN / 2)

/* Crossfade between internal read and RETURN when the MIX pot crosses
 * centre (~5 ms). It only affects the feedback, because the wet output is
 * zero in the dead zone where the switch happens. */
#define RETURN_XFADE_STEP_Q15        (32768 / 240)

extern volatile uint8_t mix_use_return;
extern volatile uint32_t clkout_trigger_tmr;

static int32_t sr_comp_line[NUM_CHAN][SEND_RETURN_LATENCY_SAMPLES];
static uint16_t sr_comp_pos[NUM_CHAN] = {0, 0};
static int32_t return_gain_q15[NUM_CHAN] = {0, 0};

/* How many samples closer to the write head the read head runs, and so how
 * long the internal stand-in for the send/return cable is. */
static uint32_t read_latency_comp(uint8_t channel)
{
	return (divmult_time[channel] >= SEND_RETURN_MIN_COMP_TIME) ? SEND_RETURN_LATENCY_SAMPLES : 0;
}

float lpf_coef;
int32_t min_vol;
float mainin_lpf[2]={0.0,0.0};
/* DC-block state: Q23.8 fixed-point (8 fractional bits) so a small DC
 * offset (~< 1 sample) can still be tracked across many ISR calls.  Alpha
 * is 1/4096 (shift 12) — corresponds to cutoff ~1.86 Hz @ 48 kHz, very
 * close to the original float coef of 1/4800. */
int32_t dcblock_state[2] = {0, 0};

enum FadeStates{
	NOT_FADING,
	WRITE_FADE_DOWN,
	WRITE_FADE_UP,
	WRITE_FADE_WRDOWN_DESTUP
};
uint8_t write_fade_state[NUM_CHAN] = {NOT_FADING,NOT_FADING};


void audio_buffer_init(void)
{
	uint32_t i;

	if (MODE_24BIT_JUMPER)
		SAMPLESIZE=4;
	else
		SAMPLESIZE=2;

	if (!ping_time)
		ping_time=0x00002000*SAMPLESIZE;


#ifdef REVERB_ENABLE
	/* Both channels are shrunk by REVERB_SDRAM_RESERVE so ch2's write head
	 * never reaches the T2 SDRAM ring at the top of its block. ch1 is
	 * matched for identical maximum loop duration. */
	loop_size[0] = LOOP_SIZE - REVERB_SDRAM_RESERVE;
	loop_size[1] = LOOP_SIZE - REVERB_SDRAM_RESERVE;
#else
	loop_size[0] = LOOP_SIZE;
	loop_size[1] = LOOP_SIZE;
#endif

	for(i=0;i<NUM_CHAN;i++){
		memory_clear(i);

		write_addr[i]=LOOP_RAM_BASE[i] + ping_time;
		read_addr[i] = LOOP_RAM_BASE[i];
		fade_dest_read_addr[i] = LOOP_RAM_BASE[i];
		fade_dest_write_addr[i] = write_addr[i];
		divmult_time[i]=ping_time;

		set_divmult_time(i);

		loop_start[i] = LOOP_RAM_BASE[i];
		loop_end[i] = LOOP_RAM_BASE[i] + loop_size[i];
		
		fractional_read_pos[i] = 0.0f;
		read_speed[i] = 1.0f;
		target_read_addr[i] = read_addr[i];
		doing_reverse_fade[i]=0;
	}

	lpf_coef = 0.0002;

	if (SAMPLESIZE==2)
	{
		min_vol = 10;
		init_compressor(1<<15, 0.75);
	}
	else
	{
		min_vol = 10 << 16;
		init_compressor(1u<<31, 0.75);  /* unsigned: 1<<31 on signed int is UB */
	}


}

uint32_t offset_samples(uint8_t channel, uint32_t base_addr, uint32_t offset, uint8_t subtract)
{
	uint32_t t_addr;

	//convert samples to addresses
	offset*=SAMPLESIZE;

	if (subtract == 0){

		t_addr = base_addr + offset;

		while (t_addr >= (LOOP_RAM_BASE[channel] + loop_size[channel]))
			t_addr = t_addr - loop_size[channel];

	} else {

		t_addr = base_addr - offset;

		while (t_addr < LOOP_RAM_BASE[channel])
			t_addr = t_addr + loop_size[channel];

	}

	if (SAMPLESIZE==2)
		t_addr = t_addr & 0xFFFFFFFE; //addresses must be even
	else
		t_addr = t_addr & 0xFFFFFFFC; //addresses must end in 00

	return (t_addr);
}


uint32_t calculate_read_addr(uint8_t channel, uint32_t new_divmult_time){
	uint32_t t_read_addr;

	t_read_addr = offset_samples(channel, write_addr[channel], new_divmult_time, 1-mode[channel][REV]);
	return (t_read_addr);
}


/*
 * varispeed_distance_samples()
 *
 * Signed delay-time error for the varispeed catch-up controller: how far the
 * read head is from where divmult_time says it should be, measured in samples
 * along the read head's own direction of travel. Positive means the read head
 * is behind and has to speed up; negative means it is ahead and has to slow
 * down.
 *
 * The ideal position is re-derived from write_addr on every call rather than
 * carried in a free-running counter. write_addr already advances at exactly 1x
 * for as long as we are recording, so this stays correct across events that
 * displace either head (reverse swap, entering/leaving INF, the end of a read
 * crossfade) — none of which a free-running counter can see, which is how the
 * read head used to end up chasing a stale target most of the way around the
 * buffer.
 */
static int32_t varispeed_distance_samples(uint8_t channel)
{
	const int32_t ring = (int32_t)loop_size[channel];

	target_read_addr[channel] = calculate_read_addr(channel, divmult_time[channel] - read_latency_comp(channel));

	int32_t distance = (int32_t)target_read_addr[channel] - (int32_t)read_addr[channel];

	/* Take the short way around the ring. The ring is loop_size, not
	 * LOOP_SIZE — they differ by REVERB_SDRAM_RESERVE (~6.8 s), which is
	 * exactly the bogus error a LOOP_SIZE correction used to invent whenever
	 * the two heads straddled the wrap point. */
	if (distance > (ring / 2))
		distance -= ring;
	else if (distance < -(ring / 2))
		distance += ring;

	if (mode[channel][REV])
		distance = -distance;

	return (distance / (int32_t)SAMPLESIZE);
}

/*
 * clock_tick_in_block()
 *
 * Which sample of the block being processed a clock tick lands on, as heard at
 * the outputs, or -1 if none does. The tick is the CLOCK OUT edge
 * (clkout_trigger_tmr wrapping), which an incoming ping resets, so it is the
 * module's tempo grid.
 *
 * clkout_trigger_tmr counts samples, but it is read at an unknown point after
 * the block ended: channel B's ISR can wait behind channel A's. The receive
 * DMA's remaining count says how far past the end of the block the codec
 * already is, so that is subtracted back out. Then the timer is lined up with
 * the audio: the block's input was captured CODEC_ADC_GROUP_DELAY earlier, and
 * the read head runs read_latency_comp() ahead of what is heard.
 */
static int32_t clock_tick_in_block(uint8_t channel, uint16_t block_len, uint32_t comp)
{
	const int32_t period = (int32_t)ping_time;
	DMA_Stream_TypeDef *rx = channel ? AUDIO_I2S2_EXT_DMA_STREAM : AUDIO_I2S3_EXT_DMA_STREAM;

	uint32_t tmr = clkout_trigger_tmr;
	uint32_t done = codec_BUFF_LEN - DMA_GetCurrDataCounter(rx);
	uint32_t frames_past_block = (done % (codec_BUFF_LEN / 2)) / 4;		/* 4 halfwords per frame */

	if (period <= (int32_t)block_len)
		return 0;		/* audio-rate clock: every block has a tick */

	/* Timer value at the last sample of this block, then the first sample
	 * where the tick falls, counting from the start of this block. */
	int32_t block_end = (int32_t)tmr - (int32_t)frames_past_block;
	int32_t j = ((int32_t)block_len - 1 + CODEC_ADC_GROUP_DELAY - (int32_t)comp) - block_end;
	j %= period;
	if (j < 0) j += period;

	return (j < (int32_t)block_len) ? j : -1;
}

/*
 * update_read_speed()
 *
 * Sets read_speed for this block so the read head tracks divmult_time while
 * keeping the repeats on the clock grid.
 *
 * - Big changes (a TIME knob step, the time switch) glide at exactly 2x or
 *   0.5x. A constant speed rescales everything evenly, so events that were
 *   on the grid stay on it. The glide starts on a clock tick, because the
 *   rescale is anchored where it starts. It also lands exactly on the target,
 *   so no leftover error keeps going round the feedback and drifting.
 * - A glide covers at most VARISPEED_GLIDE_MAX_PERIODS of the change. At that
 *   point it has run a whole number of clock periods at a whole-number speed,
 *   so the head is on the grid again. The rest of the change is a read
 *   crossfade straight to the target, which is on the grid too, so the
 *   crossfade itself needs no tick.
 * - Small errors (clock jitter) are nudged out at up to VARISPEED_NUDGE_RATE,
 *   also landing exactly.
 *
 * All of that is for the quantized TIME modes, where every change is a jump
 * between clock divisions. With the knob or the jack unquantized, TIME moves
 * continuously (sweeps, 1V/oct), and waiting for ticks would stop it bending.
 * There the head glides straight away, easing towards 2x or 0.5x at the
 * VARISPEED_INERTIA set with PING + TIME, with no glide limit. It still
 * lands exactly.
 *
 * Speed is constant within a block, so the first and last blocks of a glide
 * use a blended speed. That keeps the distance exact, and the events in those
 * blocks move by at most a few samples.
 */
static void update_read_speed(uint8_t channel, uint16_t block_len)
{
	const uint32_t comp = read_latency_comp(channel);

	/* Signed distance to the target in samples, counting the fractional
	 * position. Positive means the head is behind (the delay is too long). */
	float dist = (float)varispeed_distance_samples(channel) - fractional_read_pos[channel];
	float abs_dist = (dist < 0.0f) ? -dist : dist;

	float nudge_max = (float)(divmult_time[channel] / VARISPEED_NUDGE_MAX_FRAC);
	if (nudge_max < VARISPEED_NUDGE_MAX_SAMPLES) nudge_max = VARISPEED_NUDGE_MAX_SAMPLES;

	const uint8_t clock_locked = (mode[channel][TIMEMODE_POT] == MOD_READWRITE_TIME_Q)
							  && (mode[channel][TIMEMODE_JACK] == MOD_READWRITE_TIME_Q);

	if (!clock_locked && (abs_dist > nudge_max || vs_state[channel] == VS_GLIDING)) {
		/* Free glide: ease towards 2x or 0.5x. The budget is refilled so a
		 * switch back to quantized mid-glide carries on like a fresh glide. */
		float glide = (dist > 0.0f) ? 2.0f : 0.5f;
		float slew = param[channel][VARISPEED_INERTIA];

		if (read_speed[channel] < glide) {
			read_speed[channel] += slew;
			if (read_speed[channel] > glide) read_speed[channel] = glide;
		} else if (read_speed[channel] > glide) {
			read_speed[channel] -= slew;
			if (read_speed[channel] < glide) read_speed[channel] = glide;
		}
		vs_state[channel] = VS_GLIDING;
		glide_budget[channel] = (float)ping_time * VARISPEED_GLIDE_MAX_PERIODS;

		/* Land exactly, if this block would reach the target */
		float move = (read_speed[channel] - 1.0f) * (float)block_len;
		if ((dist > 0.0f && move >= dist) || (dist < 0.0f && move <= dist) || dist == 0.0f) {
			read_speed[channel] = 1.0f + dist / (float)block_len;
			vs_state[channel] = VS_SETTLED;
		}
		return;
	}
	if (!clock_locked && vs_state[channel] != VS_SETTLED)
		vs_state[channel] = VS_SETTLED;		/* drop a queued tick wait or crossfade */

	if (vs_state[channel] == VS_CROSSFADE) {
		if (abs_dist > nudge_max) {
			/* Same crossfade the INF/reverse code uses. The caller sees
			 * read_fade_pos and reads the destination head; the fade
			 * lands read_addr on it. */
			fade_dest_read_addr[channel] = target_read_addr[channel];
			read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
			doing_reverse_fade[channel] = 0;
			fade_queued_dest_divmult_time[channel] = 0;
			fractional_read_pos[channel] = 0.0f;
		}
		/* Otherwise TIME moved back near where the glide stopped: nudge */
		vs_state[channel] = VS_SETTLED;
		read_speed[channel] = 1.0f;
		return;
	}

	if (vs_state[channel] != VS_GLIDING) {
		if (abs_dist > nudge_max) {
			int32_t tick = clock_tick_in_block(channel, block_len, comp);
			if (tick < 0) {
				/* Hold the current delay time until the tick */
				vs_state[channel] = VS_WAITING;
				read_speed[channel] = 1.0f;
				return;
			}
			/* Tick in this block: glide from that sample on */
			vs_state[channel] = VS_GLIDING;
			glide_budget[channel] = (float)ping_time * VARISPEED_GLIDE_MAX_PERIODS;
			float glide = (dist > 0.0f) ? 2.0f : 0.5f;
			read_speed[channel] = 1.0f + (glide - 1.0f) * (float)(block_len - tick) / (float)block_len;
		}
		else {
			vs_state[channel] = VS_SETTLED;

			if (abs_dist < VARISPEED_ARRIVED) {
				/* Arrived: sit exactly on the target at 1x, so
				 * memory_read_varispeed's single-read fast path runs */
				read_addr[channel] = target_read_addr[channel];
				fractional_read_pos[channel] = 0.0f;
				read_speed[channel] = 1.0f;
				return;
			}

			float step = dist;
			float max_step = VARISPEED_NUDGE_RATE * (float)block_len;
			if (step > max_step) step = max_step;
			if (step < -max_step) step = -max_step;
			read_speed[channel] = 1.0f + step / (float)block_len;
			return;
		}
	}
	else {
		/* Keep gliding. If the target moved past us (TIME changed again
		 * mid-glide), turn round: the head is already off the grid's
		 * anchor, so there is nothing to wait for. */
		read_speed[channel] = (dist > 0.0f) ? 2.0f : 0.5f;
	}

	/* Land exactly on the target in the block that would otherwise overshoot */
	float move = (read_speed[channel] - 1.0f) * (float)block_len;
	if ((dist > 0.0f && move >= dist) || (dist < 0.0f && move <= dist) || dist == 0.0f) {
		read_speed[channel] = 1.0f + dist / (float)block_len;
		vs_state[channel] = VS_SETTLED;
		return;
	}

	/* Or stop exactly where the budget runs out, and crossfade next block */
	float abs_move = (move < 0.0f) ? -move : move;
	if (abs_move >= glide_budget[channel]) {
		read_speed[channel] = 1.0f + ((move < 0.0f) ? -glide_budget[channel] : glide_budget[channel]) / (float)block_len;
		glide_budget[channel] = 0.0f;
		vs_state[channel] = VS_CROSSFADE;
	} else
		glide_budget[channel] -= abs_move;
}

void swap_read_write(uint8_t channel){

	fade_dest_read_addr[channel] = fade_dest_write_addr[channel];
	fade_dest_write_addr[channel] = read_addr[channel];

	write_fade_pos[channel] = global_param[FAST_FADE_INCREMENT];
	write_fade_state[channel] = WRITE_FADE_WRDOWN_DESTUP;

	read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
	doing_reverse_fade[channel]=1;

	fade_queued_dest_divmult_time[channel] = 0;

}

void reverse_loop(uint8_t channel)
{
	uint32_t t;

	//When reversing in INF mode, swap the loop start/end but offset them by the FADE_SAMPLES so the crossfade stays within already recorded audio
	t=loop_start[channel];

	loop_start[channel] = offset_samples(channel, loop_end[channel], global_param[SLOW_FADE_SAMPLES], mode[channel][REV]);
	loop_end[channel] = offset_samples(channel, t, global_param[SLOW_FADE_SAMPLES], mode[channel][REV]);

	//ToDo: Add a crossfade for read head reversing direction here
	fade_dest_read_addr[channel] = read_addr[channel];

	read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
	doing_reverse_fade[channel]=1;

	fade_queued_dest_divmult_time[channel] = 0;

}

uint32_t inc_addr(uint32_t addr, uint8_t channel)
{

	if (mode[channel][REV] == 0)
	{
		addr+=SAMPLESIZE;
		if (addr >= (LOOP_RAM_BASE[channel] + loop_size[channel]))
			addr = LOOP_RAM_BASE[channel];
	}
	else
	{
		addr-=SAMPLESIZE;
		if (addr <= LOOP_RAM_BASE[channel])
			addr = LOOP_RAM_BASE[channel] + loop_size[channel] - SAMPLESIZE;
	}

	return(addr & 0xFFFFFFFE);

	//return (offset_samples(channel, addr, 1, mode[channel][REV]));
}

uint32_t dec_addr(uint32_t addr, uint8_t channel)
{

	if (mode[channel][REV] != 0)
	{
		addr+=SAMPLESIZE;
		if (addr >= (LOOP_RAM_BASE[channel] + loop_size[channel]))
			addr = LOOP_RAM_BASE[channel];
	}
	else
	{
		addr-=SAMPLESIZE;
		if (addr <= LOOP_RAM_BASE[channel])
			addr = LOOP_RAM_BASE[channel] + loop_size[channel] - 2;
	}
	return(addr & 0xFFFFFFFE);

	//return (offset_samples(channel, addr, 1, 1-mode[channel][REV]));

}

/*
 * in_between()
 *
 * Utility function to determine if address mid is in between addresses beg and end in a circular (ring) buffer.
 * To Do: draw a truth table and condense this into a few boolean logic functions
 *
 */
uint8_t in_between(uint32_t mid, uint32_t beg, uint32_t end, uint8_t reverse)
{
	uint32_t t;

	if (beg==end) //zero length, trivial case
	{
		if (mid!=beg) return(0);
		else return(1);
	}

	if (reverse) { //swap beg and end if we're reversed
		t=end;
		end=beg;
		beg=t;
	}

	if (end>beg) //not wrapped around
	{
		if ((mid>=beg) && (mid<=end)) return(1);
		else return(0);

	}
	else //end has wrapped around
	{
		if ((mid<=end) || (mid>=beg)) return(1);
		else return(0);
	}
}


/*
 * set_divmult_time()
 *
 * Changing divmult (Time knob or jack, or Ping clock speed) results in moving the read addr
 * Unless we're in INF mode, then move the loop end
 *
 * To move the read addr, we have to pay attention to the cross-fading status:
 * If we are not cross-fading the read head then
 *  -See if the new divmult_time is different than the existing one
 *   	-If so, initiate a cross-fade.
 * -Set divmult_time to the destination divmult_time
 *
 * Otherwise, if we are in the middle of a cross-fade, then just queue the new divmult_time
 *
 */

uint32_t old_divmult_time[2]={0,0};

void set_divmult_time(uint8_t channel){
	uint32_t t_divmult_time;


	if (mode[channel][PING_LOCKED])
		t_divmult_time = locked_ping_time[channel] * param[channel][TIME];
	else
		t_divmult_time = ping_time * param[channel][TIME];

	//t_divmult_time = t_divmult_time & 0xFFFFFFFC; //force it to be a multiple of 4

	// Check for valid divmult time range. Cap against the channel's own ring
	// (loop_size, which the reverb shortens) — a divmult longer than the ring
	// wraps past itself and reads as a much shorter delay.
	if (t_divmult_time > loop_size[channel]/SAMPLESIZE)
		t_divmult_time = loop_size[channel]/SAMPLESIZE;

	if (mode[channel][INF] != INF_OFF)
	{
		if (old_divmult_time[channel] != t_divmult_time){

			old_divmult_time[channel] = t_divmult_time;
			divmult_time[channel] = t_divmult_time;

			if (flag_pot_changed_revdown[TIME*2+channel])
				loop_end[channel] = offset_samples(channel, loop_start[channel], divmult_time[channel], mode[channel][REV]);
			else
				loop_start[channel] = offset_samples(channel, loop_end[channel], divmult_time[channel], 1-mode[channel][REV]);


			// If the read addr is not in between the loop start and end, then fade to the loop start
			if (!in_between(read_addr[channel], loop_start[channel], loop_end[channel],mode[channel][REV]))
			{
				if (read_fade_pos[channel] < global_param[SLOW_FADE_INCREMENT])
				{
					read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
					fade_queued_dest_divmult_time[channel] = 0;

					fade_dest_read_addr[channel] = loop_start[channel];
					reset_loopled_tmr(channel);
				}
				else
				{
					fade_queued_dest_read_addr[channel]=loop_start[channel];
				}
			}
		}
	}
	else
	{
		// INF_OFF mode: the read head varispeeds to the new delay time instead
		// of crossfading. Nothing to latch here — the catch-up controller in
		// process_audio_block_codec re-measures the error against write_addr
		// every block, so it picks the new divmult_time up on its own.
		divmult_time[channel] = t_divmult_time;
	}

}


/*
 * scroll_loop()
 *
 * Move loop_start and loop_end the same amount.
 *
 * scroll_amount specifies the amount to move it, as expressed as a fraction of the loop legnth
 * scroll_subtract flag means to subtract from loop_start and loop_end, otherwise add
 *    Thus, if loop_start is 500 and loop_end is 750, and scroll_amount is 0.4
 *    then add 0.4 * (750 - 500) = 100 to loop_start and loop_end
 *
 */

void scroll_loop(uint8_t channel, float scroll_amount, uint8_t scroll_subtract)
{
	uint32_t loop_length;
	uint32_t loop_shift;

	// Get loop length
	if (!mode[channel][REV]){
		if (loop_end[channel] > loop_start[channel])
			loop_length = loop_end[channel] - loop_start[channel];
		else
			loop_length = loop_end[channel] + LOOP_SIZE - loop_start[channel];
	}
	else
	{
		if (loop_start[channel] > loop_end[channel])
			loop_length = loop_start[channel] - loop_end[channel];
		else
			loop_length = loop_start[channel] + LOOP_SIZE - loop_end[channel];
	}

	//Calculate amount to shift
	loop_shift = (uint32_t)(scroll_amount * (float)loop_length);

	//convert the units from addresses to samples
	loop_shift = loop_shift / SAMPLESIZE;

	//Add (or subtract) to the loop points.
	loop_start[channel] = offset_samples(channel, loop_start[channel], loop_shift, scroll_subtract);
	loop_end[channel] = offset_samples(channel, loop_end[channel], loop_shift, scroll_subtract);
}


/*
 * increment_read_fade()
 *
 * If we're fading, increment the fade position
 * If we've cross-faded 100%:
 *	-Stop the cross-fade
 *	-Set read_addr to the destination
 *	-Load the next queued fade (if it exists)
 *
 */

void increment_read_fade(uint8_t channel)
{
	if (read_fade_pos[channel]>0.0)
	{
		read_fade_pos[channel] += global_param[SLOW_FADE_INCREMENT];

		if (read_fade_pos[channel] > 1.0)
		{
			read_fade_pos[channel] = 0.0;
			doing_reverse_fade[channel] = 0;
			read_addr[channel] = fade_dest_read_addr[channel];

			if (fade_queued_dest_divmult_time[channel])
			{
				divmult_time[channel] = fade_queued_dest_divmult_time[channel];
				fade_queued_dest_divmult_time[channel]=0;
				fade_dest_read_addr[channel] = calculate_read_addr(channel, divmult_time[channel]);
				read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
			}
			else if (fade_queued_dest_read_addr[channel])
			{
				fade_dest_read_addr[channel] = fade_queued_dest_read_addr[channel];
				fade_queued_dest_read_addr[channel]=0;
				read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
			}
		}
	}
}

void increment_write_fade(uint8_t channel)
{

	if (write_fade_pos[channel]>0.0){

		if (write_fade_state[channel]==WRITE_FADE_UP)
			write_fade_pos[channel] += global_param[FAST_FADE_INCREMENT];

		else if (write_fade_state[channel]==WRITE_FADE_DOWN)
			write_fade_pos[channel] += global_param[SLOW_FADE_INCREMENT];

		else if (write_fade_state[channel]==WRITE_FADE_WRDOWN_DESTUP)
			write_fade_pos[channel] += global_param[FAST_FADE_INCREMENT];

		if (write_fade_pos[channel] > 1.0)
		{
			write_fade_pos[channel] = 0.0;
			write_fade_state[channel]=NOT_FADING;
			write_addr[channel] = fade_dest_write_addr[channel];

			if (mode[channel][INF]==INF_TRANSITIONING_ON)
				mode[channel][INF]=INF_ON;

		}
	}
}


/*
 * change_inf_mode()
 *
 * Do nothing if we are write-fading
 *
 * Otherwise...
 * If INF is on, go to transition-off mode
 * Initiate a write fade-up at the read_addr
 *
 * If INF is off or transitioning off, turn it on
 * Initiate a write-fade-down at the present write_addr
 *
 *
 */
void change_inf_mode(uint8_t channel)
{
	if(write_fade_state[channel]==NOT_FADING)
	{

		flag_inf_change[channel]=0;

		if (mode[channel][INF]==INF_ON || mode[channel][INF]==INF_TRANSITIONING_ON)
		{
			mode[channel][INF] = INF_TRANSITIONING_OFF;

			write_fade_pos[channel] = global_param[FAST_FADE_INCREMENT];
			write_fade_state[channel]=WRITE_FADE_UP;
			fade_dest_write_addr[channel] = read_addr[channel];
		}

		else
		{
			//Don't change the loop start/end if we hit INF off recently (recent enough that we're still T_OFF)
			//This is because the read and write pointers are in the same spot
			if (mode[channel][INF] != INF_TRANSITIONING_OFF)
			{
				reset_loopled_tmr(channel);

				loop_start[channel] = fade_dest_read_addr[channel]; //use the dest because if we happen to be fading the read head when we hit inf (e.g. changing divmult time) then we should loop between the new points since divmult_time (used in the next line) corresponds with the dest
				//The read head runs read_latency_comp() ahead of the delay time (see SEND_RETURN_LATENCY_SAMPLES), so step back to where the delay time really starts; otherwise the loop's tail would run past the write head into unwritten audio
				loop_start[channel] = offset_samples(channel, loop_start[channel], read_latency_comp(channel), 1-mode[channel][REV]);
				loop_end[channel] = offset_samples(channel, loop_start[channel], divmult_time[channel], mode[channel][REV]);
			}
			write_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
			write_fade_state[channel]=WRITE_FADE_DOWN;
			fade_dest_write_addr[channel] = write_addr[channel];

			mode[channel][INF] = INF_TRANSITIONING_ON;

		}

	}
}

/*
 * abs_diff()
 *
 * returns the absolute difference between uint32_t values
 *
 */
uint32_t abs_diff(uint32_t a1, uint32_t a2)
{
	if (a1>a2) return (a1-a2);
	else return (a2-a1);
}


enum AutoMute_States{
	MUTED,
	FADING_DOWN,
	FADING_UP,
	UNMUTED
};
/*
 * process_audio_block_codec()
 *
 * Process the audio
 * This is called by the RX DMA interrupt for each codec
 *
 * parameter sz is codec_BUFF_LEN/4 samples per channel (currently 8)
 *
 */
#include "diag_log.h"

void process_audio_block_codec(int16_t *src, int16_t *dst, int16_t sz, uint8_t channel)
{
	uint32_t _isr_t0 = DWT->CYCCNT;
	static uint32_t mute_on_boot_ctr=96000;
	static uint8_t auto_muting_main_state[NUM_CHAN]={0,0};
	static float auto_muting_main_fade[NUM_CHAN]={0,0};

	/* Channel B's per-sample contribution to the mono reverb input, handed
	 * across to channel A's ISR, which is the one that pushes into the reverb.
	 * Sized to the largest block this function is ever called with
	 * (codec_BUFF_LEN/4; the loop below runs sz/2 = 16 iterations today). */
	static int16_t chB_reverb_contrib[codec_BUFF_LEN/4]={0};

	/* (Snapshot removed — reverted to direct reads from volatile DMA buffer
	 * to match HEAD behaviour while we hunt the input-leak bug.) */

	uint32_t last_read_block_addr;

	int32_t mainin, mix, dry, wr, rd;
	int32_t regen;          /* was float — now Q-format int from Q15 mul */
	int32_t mainin_atten;   /* was float — now Q-format int from Q15 mul */
	int32_t auxin;
	int32_t auxout;
	int32_t loop_rd;        /* delay read as the loop sees it: internal, or via RETURN */

	/* Pre-compute Q15 versions of params used in the inner loop, so we do
	 * one VCVT+VMUL per param up front instead of per sample × 4 iters. */
	const int32_t regen_q15   = (int32_t)(param[channel][REGEN]   * 32768.0f);
	const int32_t level_q15   = (int32_t)(param[channel][LEVEL]   * 32768.0f);
	const int32_t mix_dry_q15 = (int32_t)(param[channel][MIX_DRY] * 32768.0f);
	const int32_t mix_wet_q15 = (int32_t)(param[channel][MIX_WET] * 32768.0f);
	/* Reverb SEND gain from right MIX pot (equal-power LUT in params.c).
	 * Scales the audio fed into the reverb via push_sample. The reverb's
	 * output is always mixed unscaled into the final delay-mix, so this
	 * knob controls how hard the reverb is driven (and therefore the
	 * audible reverb amount, but only via the tail's own input level). */
	const int32_t send_q15 = (int32_t)(reverb_send * 32768.0f);
	/* Top 10% of the right MIX pot fades the dry+delay path out (set by
	 * params.c), so at max the output is 100% wet reverb. */
	const int32_t dry_gain_q15 = (int32_t)(reverb_dry_gain * 32768.0f);
	/* Send/Return routing for this block: which side of centre the MIX pot
	 * is on, and how long the internal stand-in for the cable is. */
	const int32_t return_target_q15 = mix_use_return ? 32768 : 0;
	const uint32_t comp = read_latency_comp(channel);

	uint16_t i,t;
	uint16_t topbyte, bottombyte;

	int32_t rd_buff[codec_BUFF_LEN/4];
	int32_t rd_buff_dest[codec_BUFF_LEN/4];
	int32_t wr_buff[codec_BUFF_LEN/4];

	uint32_t crossed_start_fade_addr;
	uint32_t start_fade_addr;

	int32_t dummy;

	uint32_t t32;


	//Sanity check to made sure the read_addr is inside the loop.
	//We shouldn't have to do this. The likely reason the read_addr escapes the loop
	//is that it passes the start_fade_addr and triggers the crossed_start_fade_addr block,
	//while at the same time already in the middle of a cross fade due to change in divmult_time or reverse
	//What to do? If we queue to crossed_start_fade_addr fade then we risk overflowing out of the loop
	//We could set start_fade_addr to be much earlier than the loop_end (by a factor of 2?) so that we won't go
	//past the loop_end even if we have to do two cross fades. Of course, this means usually our loop will be earlier by
	//one crossfade period, maybe 3ms or so. This seems acceptable, but a better solution could be desired.


	if ((mode[channel][INF]==INF_ON || mode[channel][INF]==INF_TRANSITIONING_OFF || mode[channel][INF]==INF_TRANSITIONING_ON)
			&& (!in_between(read_addr[channel], loop_start[channel], loop_end[channel], mode[channel][REV])))
	{
		if (read_fade_pos[channel] < global_param[SLOW_FADE_INCREMENT])
		{
			read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];
			fade_queued_dest_divmult_time[channel] = 0;

			fade_dest_read_addr[channel] = loop_start[channel];

			reset_loopled_tmr(channel);

		}
	}



	//For short periods (audio rate), disble crossfading before the end of the loop
	if (divmult_time[channel] < (global_param[SLOW_FADE_SAMPLES]))
		start_fade_addr = loop_end[channel];
	else
		start_fade_addr = offset_samples(channel, loop_end[channel], global_param[SLOW_FADE_SAMPLES] / SAMPLESIZE, 1-mode[channel][REV]);

	// Read from memory into the main read buffer:
	// This in/decrements the read_addr based on the REV mode,
	// and also based on doing_reverse_fade (in which case the read_addr goes the opposite direction as REV mode would indicate)
	// crossed_start_fade_addr is true if read addr crosses the end of the loop, in which case we need to reset it to the beginning of the loop.
	// If doing_reverse_fade is true, then we should read in the opposite direction as mode[][REV] dictates (this is because we just
	// reversed direction, so we should continue reading from rd_buff in the same direction (which is now !REV),
	// and cross fade towards dest_rd_buff being read in the direction of REV

#ifdef DIAG_DELAY_SDRAM_TIMING
	uint32_t _rd_t0 = DWT->CYCCNT;
#endif
	if (mode[channel][INF] == INF_OFF) {
		if (read_fade_pos[channel] > 0.0f) {
			// A read crossfade is in flight (reverse swap, or the hand-off out
			// of INF). Both heads are deliberately in transit, so the error
			// against write_addr is meaningless until the fade lands — and
			// holding 1x keeps rd_buff and rd_buff_dest sample-aligned so they
			// can actually be mixed. Catch-up resumes on the block after
			// increment_read_fade drops read_addr on its destination.
			read_speed[channel] = 1.0f;
			fractional_read_pos[channel] = 0.0f;
		} else {
			update_read_speed(channel, sz/2);
		}

		crossed_start_fade_addr = memory_read_varispeed(read_addr, &fractional_read_pos[channel], channel, rd_buff, sz/2, read_speed[channel], start_fade_addr, doing_reverse_fade[channel]);
	} else {
		// Freeze modes: use original memory_read. Drop any queued or
		// half-done glide; the head is re-measured when INF ends.
		vs_state[channel] = VS_SETTLED;
		crossed_start_fade_addr = memory_read(read_addr, channel, rd_buff, sz/2, start_fade_addr, doing_reverse_fade[channel]);
	}


	if (mode[channel][INF]!=INF_OFF && crossed_start_fade_addr)
	{
		reset_loopled_tmr(channel);

		if (divmult_time[channel] < (global_param[SLOW_FADE_SAMPLES]))
		{
			read_addr[channel]=loop_start[channel];
			read_fade_pos[channel] = 0.0;

			//Issue: is it necessary to set this below?
			fade_dest_read_addr[channel] = offset_samples(channel, read_addr[channel], sz/SAMPLESIZE, 1-mode[channel][REV]);

			if (mode[channel][INF]==INF_TRANSITIONING_OFF)
			{
				mode[channel][INF]=INF_OFF;
			}

		}
		else
		{
			read_fade_pos[channel] = global_param[SLOW_FADE_INCREMENT];

			//Issue: clearing a queued divmult time?
			fade_queued_dest_divmult_time[channel]=0;

			//Start fading from before the loop
			//We have to add in sz because read_addr has already been incremented by sz since a block was just read
			if (mode[channel][REV])
				fade_dest_read_addr[channel] = offset_samples(channel, read_addr[channel], ((loop_start[channel]-loop_end[channel])+sz)/SAMPLESIZE, 0);
			else
				fade_dest_read_addr[channel] = offset_samples(channel, read_addr[channel], ((loop_end[channel]-loop_start[channel])+sz)/SAMPLESIZE, 1);

			if (mode[channel][INF]==INF_TRANSITIONING_OFF)
			{
				mode[channel][INF]=INF_OFF;
			}

		}



	}

	// Read crossfade destination buffer. Needed for the freeze modes, and in
	// delay mode whenever a read fade is in flight — reversing (swap_read_write)
	// and the hand-off out of INF both set one up, and without the destination
	// buffer the fade never happens: read_addr just jumps to fade_dest_read_addr
	// when the fade "completes".
	if (mode[channel][INF] != INF_OFF || read_fade_pos[channel] > 0.0f) {
		memory_read(fade_dest_read_addr, channel, rd_buff_dest, sz/2, 0, 0 /* + mode[channel][CONTINUOUS_REVERSE]*/);
	}

#ifdef DIAG_DELAY_SDRAM_TIMING
	diag_log(DIAG_EVT_ISR_SDRAM_READ, DWT->CYCCNT - _rd_t0);
#endif

	for (i=0;i<(sz/2);i++){

		// Split incoming stereo audio into the two channels: Left=>Main input (clean), Right=>Aux Input

		if (SAMPLESIZE==2){
			mainin = (*src++) /*+ CODEC_ADC_CALIBRATION_DCOFFSET[channel+0]*/;
			dummy=*src++;

			auxin = (*src++) /*+ CODEC_ADC_CALIBRATION_DCOFFSET[channel+2]*/;
			dummy=*src++;
		}
		else
		{
			/* Cast topbyte to uint32_t before the 16-bit shift — uint16_t
			 * promotes to (signed) int, and if bit 15 is set the shifted
			 * result lands in the sign bit of int, which is UB. */
			topbyte = (uint16_t)(*src++);
			bottombyte = (uint16_t)(*src++);
			mainin = (int32_t)(((uint32_t)topbyte << 16) | (uint16_t)bottombyte);

			topbyte = (uint16_t)(*src++);
			bottombyte = (uint16_t)(*src++);
			auxin = (int32_t)(((uint32_t)topbyte << 16) | (uint16_t)bottombyte);
		}

		if (mute_on_boot_ctr)
		{
			mute_on_boot_ctr--;
			mainin=0;
			auxin=0;
		}

		if (global_mode[AUTO_MUTE]){
			mainin_lpf[channel] = (mainin_lpf[channel]*(1.0f-lpf_coef)) + (((mainin>0)?mainin:(-1*mainin))*lpf_coef);
			if (mainin_lpf[channel]<min_vol && (auto_muting_main_state[channel] == FADING_UP || auto_muting_main_state[channel]==UNMUTED))
				auto_muting_main_state[channel] =  FADING_DOWN;
			if (mainin_lpf[channel]>=min_vol && (auto_muting_main_state[channel] == FADING_DOWN || auto_muting_main_state[channel]==MUTED))
				auto_muting_main_state[channel] =  FADING_UP;

			if (auto_muting_main_state[channel] == FADING_DOWN)
				auto_muting_main_fade[channel] -= AUTO_MUTE_DECAY;
			else if (auto_muting_main_state[channel] == FADING_UP)
				auto_muting_main_fade[channel] += AUTO_MUTE_ATTACK;
			if (auto_muting_main_fade[channel] <= 0.0f) {
				auto_muting_main_fade[channel] = 0.0f;
				auto_muting_main_state[channel] = MUTED;
			} else if (auto_muting_main_fade[channel] >= 1.0f) {
				auto_muting_main_fade[channel] = 1.0f;
				auto_muting_main_state[channel] = UNMUTED;
			}
			if (auto_muting_main_state[channel] == MUTED)
				mainin = 0;
			else if (auto_muting_main_state[channel] != UNMUTED)
				mainin = (float)mainin * auto_muting_main_fade[channel];

		}


		// The Dry signal is just the clean signal, without any attenuation from LEVEL
		dry = mainin;


		// Read from the loop and save this value so we can output it to the Delay Out jack
		if (mode[channel][INF] == INF_OFF && read_fade_pos[channel] <= 0.0f) {
			// Varispeed mode, not fading: use interpolated samples directly
			rd = rd_buff[i];
		} else {
			// Crossfade between the current and destination read heads
			t = (uint16_t)(4095.0f * read_fade_pos[channel]);
			asm("usat %[dst], #12, %[src]" : [dst] "=r" (t) : [src] "r" (t));
			rd = ((float)rd_buff[i] * epp_lut[t]) + ((float)rd_buff_dest[i] * epp_lut[4095-t]);
		}




		if (global_mode[SOFTCLIP])
			rd = compress(rd);

		if (SAMPLESIZE==2)
			asm("ssat %[dst], #16, %[src]" : [dst] "=r" (rd) : [src] "r" (rd));

		/* Send/Return insert in the delay loop (see SEND_RETURN_LATENCY_SAMPLES).
		 * SEND always carries the buffer read. What the rest of the loop
		 * treats as "the read" (loop_rd, used for regen and the wet output)
		 * is either that same read, delayed by the internal stand-in for the
		 * cable, or the RETURN jack. Which one depends on the side of the
		 * MIX pot, with a short crossfade when it changes. */
		auxout = rd;

		if (comp) {
			int32_t delayed = sr_comp_line[channel][sr_comp_pos[channel]];
			sr_comp_line[channel][sr_comp_pos[channel]] = rd;
			if (++sr_comp_pos[channel] >= SEND_RETURN_LATENCY_SAMPLES)
				sr_comp_pos[channel] = 0;
			rd = delayed;
		}

		if (return_gain_q15[channel] != return_target_q15) {
			if (return_gain_q15[channel] < return_target_q15) {
				return_gain_q15[channel] += RETURN_XFADE_STEP_Q15;
				if (return_gain_q15[channel] > return_target_q15)
					return_gain_q15[channel] = return_target_q15;
			} else {
				return_gain_q15[channel] -= RETURN_XFADE_STEP_Q15;
				if (return_gain_q15[channel] < return_target_q15)
					return_gain_q15[channel] = return_target_q15;
			}
		}

		if (return_gain_q15[channel] == 32768)
			loop_rd = auxin;
		else if (return_gain_q15[channel] == 0)
			loop_rd = rd;
		else
			loop_rd = rd + (((auxin - rd) * return_gain_q15[channel]) >> 15);

		/* Integer Q15-format multiplies replace the per-sample float math.
		 * regen_q15 / level_q15 / mix_dry_q15 / mix_wet_q15 are pre-computed
		 * once per ISR above. */
		regen        = (loop_rd * regen_q15) >> 15;
		mainin_atten = (mainin  * level_q15) >> 15;

		wr     = regen + mainin_atten;

		/* DC blocker as a 1-pole IIR in Q23.8 fixed point.
		 *   state += ((wr << 8) - state) >> 12         alpha ≈ 1/4096
		 *   wr    -= state >> 8                        subtract DC estimate
		 * Cutoff ~1.86 Hz @ 48 kHz, close to the original float 1/4800.
		 */
		if (global_mode[RUNAWAYDC_BLOCK])
		{
			/* wr may be negative here (it's a sum of three signed values
			 * before the SSAT on line below). Left-shifting a negative
			 * signed int is UB; cast through unsigned to get well-defined
			 * bitwise behaviour that GCC also produces less defensively. */
			int32_t err = (int32_t)((uint32_t)wr << 8) - dcblock_state[channel];
			dcblock_state[channel] += err >> 12;
			wr -= dcblock_state[channel] >> 8;
		}

		if (global_mode[SOFTCLIP])
			wr = compress(wr);
		else if (SAMPLESIZE==2)
			asm("ssat %[dst], #16, %[src]" : [dst] "=r" (wr) : [src] "r" (wr));

		// Wet/dry mix in Q15.
		mix = ((dry * mix_dry_q15) + (loop_rd * mix_wet_q15)) >> 15;

		if (global_mode[SOFTCLIP])
			mix = compress(mix);

		else if (SAMPLESIZE==2)
			asm("ssat %[dst], #16, %[src]" : [dst] "=r" (mix) : [src] "r" (mix));

#ifdef REVERB_ENABLE
		/* --- Reverb: fed by a mono sum of both channels ---
		 * Right MIX scales the audio FED INTO the reverb. The reverb's stereo
		 * output is then summed unscaled into the per-channel mix. So the
		 * right MIX controls how hard the reverb is driven; the tail you hear
		 * scales with that drive, but the output mix is never silenced. */
		{
			int16_t rev_s;
			/* This channel's contribution to the mono reverb input. */
			int32_t rev_contrib = mix;

			if (channel == 0) {
				/* Mono-sum both channels, averaged so correlated material
				 * keeps the level the reverb saw when it was fed channel A
				 * alone.
				 *
				 * Channel B's half is handed over from its own ISR. The two
				 * share an NVIC preemption priority, so neither can interrupt
				 * the other and the buffer can never be read half-written.
				 * Which one runs first is not guaranteed though: they are
				 * separate DMA interrupts, and the sub-priority tie-break
				 * (A = 0, B = 1) only decides the order when both happen to
				 * pend on the same cycle. So B's contribution is either from
				 * this block or the previous one — at most 16 samples,
				 * ~0.33 ms, of skew on one half of a mono send into a diffuse
				 * reverb, which is well below audibility either way. */
				int32_t to_reverb = ((rev_contrib + (int32_t)chB_reverb_contrib[i]) >> 1);
				to_reverb = (to_reverb * send_q15) >> 15;
				if (SAMPLESIZE==2)
					asm("ssat %[dst], #16, %[src]" : [dst] "=r" (to_reverb) : [src] "r" (to_reverb));
				velvet_reverb_push_sample((int16_t)to_reverb);
				rev_s = velvet_reverb_out_left();
			} else {
				chB_reverb_contrib[i] = (int16_t)rev_contrib;
				rev_s = velvet_reverb_out_right();
			}
			/* Fade the dry+delay path out over the top 10% of the right MIX
			 * pot. Reverb is fed the un-faded mix above, so at max knob the
			 * output is 100% wet reverb (mix→0, rev_s unchanged). */
			mix = (mix * dry_gain_q15) >> 15;
			mix += rev_s;
			if (SAMPLESIZE==2)
				asm("ssat %[dst], #16, %[src]" : [dst] "=r" (mix) : [src] "r" (mix));
		}
#endif

		if (global_mode[CALIBRATE])
		{
			*dst++ = CODEC_DAC_CALIBRATION_DCOFFSET[0+channel];
			*dst++ = 0;

			*dst++ = CODEC_DAC_CALIBRATION_DCOFFSET[2+channel];
			*dst++ = 0;
		}
		else
		{

#ifdef DEBUG_POTADC_TO_CODEC
			*dst++ = potadc_buffer[channel+0]*4;
			*dst++ = 0;

			if (TIMESW_CH1==SWITCH_CENTER) *dst++ = potadc_buffer[channel+2]*4;
			else if (TIMESW_CH1==SWITCH_UP) *dst++ = potadc_buffer[channel+4]*4;
			else *dst++ = potadc_buffer[channel+6]*4;
			*dst++ = 0;
#else
#ifdef DEBUG_CVADC_TO_CODEC
			*dst++ = potadc_buffer[channel+2]*4;
			*dst++ = 0;

			if (TIMESW_CH1==SWITCH_CENTER) *dst++ = cvadc_buffer[channel+4]*4;
			else if (TIMESW_CH1==SWITCH_UP) *dst++ = cvadc_buffer[channel+0]*4;
			else *dst++ = cvadc_buffer[channel+2]*4;
			*dst++ = 0;

#else

			if (SAMPLESIZE==2){
				int32_t main_out = mix + CODEC_DAC_CALIBRATION_DCOFFSET[0+channel];
				asm("ssat %[d], #16, %[s]" : [d] "=r" (main_out) : [s] "r" (main_out));
				*dst++ = (int16_t)main_out;
				*dst++ = 0;

				//Send out — channel 1's send carries the FSK diag stream
				//when DIAG_FSK_ENABLE is on and diag_log_enabled is true.
#ifdef DIAG_FSK_ENABLE
				if (diag_log_enabled && channel == 1) {
					*dst++ = diag_fsk_next_sample();
					*dst++ = 0;
				} else
#endif
				{
					int32_t send_out = auxout + CODEC_DAC_CALIBRATION_DCOFFSET[2+channel];
					asm("ssat %[d], #16, %[s]" : [d] "=r" (send_out) : [s] "r" (send_out));
					*dst++ = (int16_t)send_out;
					*dst++ = 0;
				}
			}
			else
			{
				//Main out
				*dst++ = (int16_t)(mix>>16) + (int16_t)CODEC_DAC_CALIBRATION_DCOFFSET[0+channel];
				*dst++ = (int16_t)(mix & 0x0000FF00);

				//Send out
				*dst++ = (int16_t)(auxout>>16) + (int16_t)CODEC_DAC_CALIBRATION_DCOFFSET[2+channel];
				*dst++ = (int16_t)(auxout & 0x0000FF00);
			}
#endif
#endif
		}

		wr_buff[i]=wr;

	}

	//Write a block to memory

#ifdef DIAG_DELAY_SDRAM_TIMING
	uint32_t _wr_t0 = DWT->CYCCNT;
#endif
	if (mode[channel][INF] == INF_OFF || mode[channel][INF]==INF_TRANSITIONING_OFF)
	{

		if (write_fade_state[channel] == WRITE_FADE_WRDOWN_DESTUP)
		{
			memory_fade_write(fade_dest_write_addr, channel, wr_buff, sz/2, 0, write_fade_pos[channel]);
			memory_fade_write(write_addr, channel, wr_buff, sz/2, 1, 1.0-write_fade_pos[channel]); //write in the opposite direction of [REV]
		}
		else if (write_fade_state[channel] == WRITE_FADE_UP)
		{
			memory_fade_write(fade_dest_write_addr, channel, wr_buff, sz/2, 0, write_fade_pos[channel]);
			write_addr[channel] = fade_dest_write_addr[channel];
		}
		else/* if (write_fade_pos[channel] < global_param[SLOW_FADE_INCREMENT])*/
		{
			memory_write(write_addr, channel, wr_buff, sz/2, 0);
			fade_dest_write_addr[channel] = write_addr[channel];
		}

	}
	else if (mode[channel][INF]==INF_TRANSITIONING_ON)
	{
		if (write_fade_state[channel]==WRITE_FADE_DOWN)
		{
			memory_fade_write(fade_dest_write_addr, channel, wr_buff, sz/2, 0, 1.0-write_fade_pos[channel]);
			write_addr[channel] = fade_dest_write_addr[channel];
		}
	}

#ifdef DIAG_DELAY_SDRAM_TIMING
	diag_log(DIAG_EVT_ISR_SDRAM_WRITE, DWT->CYCCNT - _wr_t0);
#endif


#ifdef ALLOW_CONT_REVERSE

	if (mode[channel][CONTINUOUS_REVERSE])
	{
		if (abs_diff(write_addr[channel], read_addr[channel]) < 960) //10ms pulse
		{
			set_loop_led(channel, 1);
			if (channel==0) CLKOUT1_ON;
			else CLKOUT2_ON;
		}
		else
		{
			set_loop_led(channel, 0);
			if (channel==0) CLKOUT1_OFF;
			else CLKOUT2_OFF;
		}
	}
#endif


	increment_read_fade(channel);
	increment_write_fade(channel);

	{
		uint32_t _isr_dt = DWT->CYCCNT - _isr_t0;
		diag_isr_cycles += _isr_dt;   /* so main-loop stage timers can subtract ISR preemption */
		diag_log(channel == 0 ? DIAG_EVT_AUDIOISR_CH0 : DIAG_EVT_AUDIOISR_CH1, _isr_dt);
	}
}
