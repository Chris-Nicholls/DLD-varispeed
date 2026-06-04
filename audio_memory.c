/*
 * audio_memory.c - audio buffer SDRAM access functions
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

#include "globals.h"
#include "audio_memory.h"
#include "looping_delay.h"
#include "params.h"

extern const uint32_t LOOP_RAM_BASE[NUM_CHAN];

extern uint8_t SAMPLESIZE;

extern uint32_t target_read_addr[];

extern uint32_t loop_size[NUM_CHAN];
extern uint8_t mode[NUM_CHAN][NUM_CHAN_MODES];

/* May-alias typedef for the fused 32-bit SDRAM read in the varispeed slow
 * path. The same SDRAM cells are written via int16_t* by memory_write, so a
 * plain (uint32_t *) cast violates strict aliasing. may_alias tells GCC the
 * accesses can alias and disables the optimization. */
typedef uint32_t u32_alias __attribute__((may_alias));

// From looping_delay.c - address increment/decrement functions that handle REV mode
extern uint32_t inc_addr(uint32_t addr, uint8_t channel);
extern uint32_t dec_addr(uint32_t addr, uint8_t channel);

void memory_clear(uint8_t channel)
{

	uint32_t i;

	//Takes 700ms to clear the channel buffer in 32-bit chunks, roughly 83ns per write
	for(i = LOOP_RAM_BASE[channel]; i < (LOOP_RAM_BASE[channel] + LOOP_SIZE); i += 4)
			*((uint32_t *)i) = 0x00000000;


}

uint32_t memory_read(uint32_t *addr, uint8_t channel, int32_t *rd_buff, uint8_t num_samples, uint32_t loop_addr, uint8_t decrement){
	uint8_t i;
	uint32_t heads_crossed=0;

	//Loop of 8 takes 2.5us
	//read from SDRAM. first one takes 200us, subsequent reads take 50ns
	for (i=0;i<num_samples;i++){

		//Enforce valid addr range
		if ((addr[channel]<SDRAM_BASE) || (addr[channel] > (SDRAM_BASE + SDRAM_SIZE)))
		addr[channel]=SDRAM_BASE;

		//even addresses only
		addr[channel] = (addr[channel] & 0xFFFFFFFE);

		while(FMC_GetFlagStatus(FMC_Bank2_SDRAM, FMC_FLAG_Busy) != RESET){;}

		if (SAMPLESIZE==2)
			rd_buff[i] = *((int16_t *)(addr[channel]));
		else
			rd_buff[i] = *((int32_t *)(addr[channel]));

		if (decrement)
			addr[channel] = dec_addr(addr[channel], channel);
		else
			addr[channel] = inc_addr(addr[channel], channel);

		if (addr[channel]==loop_addr) heads_crossed=1;

	}

	return(heads_crossed);
}

uint32_t memory_read_varispeed(uint32_t *addr, float *frac_pos, uint8_t channel, int32_t *rd_buff, uint8_t num_samples, float speed, uint32_t loop_addr, uint8_t decrement)
{
	uint8_t i;
	uint32_t heads_crossed = 0;

	/* Fast path: exact 1x playback on an integer sample boundary needs no
	 * interpolation. Keep this optimization inside the memory helper so the
	 * caller's varispeed state machine stays intact. */
	if (speed > 0.999f && speed < 1.001f && *frac_pos > -0.001f && *frac_pos < 0.001f) {
		for (i = 0; i < num_samples; i++) {
			if (decrement)
				target_read_addr[channel] = dec_addr(target_read_addr[channel], channel);
			else
				target_read_addr[channel] = inc_addr(target_read_addr[channel], channel);

			if ((addr[channel] < SDRAM_BASE) || (addr[channel] > (SDRAM_BASE + SDRAM_SIZE)))
				addr[channel] = SDRAM_BASE;

			addr[channel] = (addr[channel] & 0xFFFFFFFE);

			while(FMC_GetFlagStatus(FMC_Bank2_SDRAM, FMC_FLAG_Busy) != RESET){;}

			if (SAMPLESIZE == 2)
				rd_buff[i] = *((int16_t *)(addr[channel]));
			else
				rd_buff[i] = *((int32_t *)(addr[channel]));

			if (decrement)
				addr[channel] = dec_addr(addr[channel], channel);
			else
				addr[channel] = inc_addr(addr[channel], channel);

			if (addr[channel] == loop_addr)
				heads_crossed = 1;
		}

		*frac_pos = 0.0f;
		return heads_crossed;
	}

	/* Slow path: linear interp with arbitrary speed. Hoist direction,
	 * loop bounds, fast 32-bit fused read, and Q24 frac math out of the
	 * inner loop so each sample is mostly two SDRAM reads + a multiply. */
	const uint32_t loop_base = LOOP_RAM_BASE[channel];
	const uint32_t loop_end  = loop_base + loop_size[channel];
	const int reversed       = (mode[channel][REV] != 0) ^ (decrement != 0);
	const int32_t  step      = reversed ? -(int32_t)SAMPLESIZE : (int32_t)SAMPLESIZE;
	const int32_t  ss        = (int32_t)SAMPLESIZE;
	const int      is16      = (SAMPLESIZE == 2);

	/* Q24 fixed-point fractional position. Caller passes a float frac_pos in
	 * [0, 1); convert once on entry and once on exit. The inner loop's frac
	 * advance is a plain int32 add and the interp lerp is an int32 mul/shift. */
	int32_t frac_q24  = (int32_t)((*frac_pos) * 16777216.0f);
	int32_t speed_q24 = (int32_t)(speed * 16777216.0f);

	uint32_t a = addr[channel] & 0xFFFFFFFEu;

	for (i = 0; i < num_samples; i++) {
		/* Target advance: keep using inc/dec_addr — only one call per output
		 * sample, and target_read_addr is global state callers depend on. */
		if (decrement)
			target_read_addr[channel] = dec_addr(target_read_addr[channel], channel);
		else
			target_read_addr[channel] = inc_addr(target_read_addr[channel], channel);

		int32_t sample0, sample1;

		/* Compute next_addr inline. Common case: not at a wrap edge.
		 * For step<0 we match dec_addr's `addr <= loop_base` semantics: the
		 * sample at loop_base itself is never visited in REV mode. */
		uint32_t na = (uint32_t)((int32_t)a + step);
		int at_boundary;
		if (step > 0)
			at_boundary = (na >= loop_end);
		else
			at_boundary = (na <= loop_base);

		if (__builtin_expect(at_boundary, 0)) {
			/* Wrap: fall through to inc/dec_addr's full semantics. */
			if (decrement)
				na = dec_addr(a, channel);
			else
				na = inc_addr(a, channel);
			na &= 0xFFFFFFFEu;

			if (is16) {
				sample0 = *((int16_t *)a);
				sample1 = *((int16_t *)na);
			} else {
				sample0 = *((int32_t *)a);
				sample1 = *((int32_t *)na);
			}
		} else {
			/* Fast contiguous read. For 16-bit samples with step=+2, fuse
			 * both reads into a single 32-bit load — one AHB transaction
			 * instead of two. */
			if (is16) {
				if (step == 2) {
					uint32_t w = *((volatile u32_alias *)a);
					sample0 = (int32_t)(int16_t)(w & 0xFFFFu);
					sample1 = (int32_t)(int16_t)(w >> 16);
				} else {
					/* step == -2: cheaper to do two halfword reads than to
					 * decode a fused word starting at na. */
					sample0 = *((int16_t *)a);
					sample1 = *((int16_t *)na);
				}
			} else {
				sample0 = *((int32_t *)a);
				sample1 = *((int32_t *)na);
			}
		}

		/* Q24 lerp: rd = s0 + ((s1 - s0) * frac_q24) >> 24. With 16-bit
		 * samples the difference fits easily in 17 bits; the multiply is a
		 * single-cycle SMULL. */
		rd_buff[i] = sample0 + (int32_t)(((int64_t)(sample1 - sample0) * frac_q24) >> 24);

		frac_q24 += speed_q24;

		/* Advance addr by however many integer steps frac rolled over.
		 * Inline the wrap; only call inc/dec_addr at a true boundary. */
		while (frac_q24 >= (1 << 24)) {
			frac_q24 -= (1 << 24);

			uint32_t na2 = (uint32_t)((int32_t)a + step);
			int wrap;
			if (step > 0)
				wrap = (na2 >= loop_end);
			else
				wrap = (na2 <= loop_base);

			if (__builtin_expect(wrap, 0)) {
				if (decrement)
					a = dec_addr(a, channel);
				else
					a = inc_addr(a, channel);
				a &= 0xFFFFFFFEu;
			} else {
				a = na2;
			}

			if (a == loop_addr)
				heads_crossed = 1;
		}
	}

	addr[channel] = a;
	/* Convert Q24 back to float for the caller's state. */
	*frac_pos = (float)frac_q24 * (1.0f / 16777216.0f);

	return heads_crossed;
}

void memory_write(uint32_t *addr, uint8_t channel, int32_t *wr_buff, uint8_t num_samples, uint8_t decrement)
{
	uint8_t i;

	for (i=0;i<num_samples;i++){

		//Enforce valid addr range
		if ((addr[channel]<SDRAM_BASE) || (addr[channel] > (SDRAM_BASE + SDRAM_SIZE)))
			addr[channel]=SDRAM_BASE;

		//even addresses only
		addr[channel] = (addr[channel] & 0xFFFFFFFE);

		while(FMC_GetFlagStatus(FMC_Bank2_SDRAM, FMC_FLAG_Busy) != RESET){;}

		if (SAMPLESIZE==2)
			*((int16_t *)addr[channel]) = wr_buff[i];
		else
			*((int32_t *)addr[channel]) = wr_buff[i];

		if (decrement)
			addr[channel] = dec_addr(addr[channel], channel);
		else
			addr[channel] = inc_addr(addr[channel], channel);


	}

}

//
// reads from the addr, and mixes that value with the value in wr_buff
// fade=1.0 means write 100% wr_buff and 0% read.
// fade=0.5 means write 50% wr_buff and 50% read.
// fade=0.0 means write 0% wr_buff and 100% read.
//
void memory_fade_write(uint32_t *addr, uint8_t channel, int32_t *wr_buff, uint8_t num_samples, uint8_t decrement, float fade){
	uint8_t i;
	int32_t rd;
	int32_t mix;

	for (i=0;i<num_samples;i++){

		while(FMC_GetFlagStatus(FMC_Bank2_SDRAM, FMC_FLAG_Busy) != RESET){;}

		//Enforce valid addr range
		if ((addr[channel]<SDRAM_BASE) || (addr[channel] > (SDRAM_BASE + SDRAM_SIZE)))
			addr[channel]=SDRAM_BASE;

		//even addresses only
		addr[channel] = (addr[channel] & 0xFFFFFFFE);

		//read from address
		if (SAMPLESIZE==2)
			rd = *((int16_t *)(addr[channel]));
		else
			rd = *((int32_t *)(addr[channel]));

		mix = ((float)wr_buff[i] * fade) + ((float)rd * (1.0-fade));

		while(FMC_GetFlagStatus(FMC_Bank2_SDRAM, FMC_FLAG_Busy) != RESET){;}

		if (SAMPLESIZE==2)
			*((int16_t *)addr[channel]) = mix;
		else
			*((int32_t *)addr[channel]) = mix;

		if (decrement)
			addr[channel] = dec_addr(addr[channel], channel);
		else
			addr[channel] = inc_addr(addr[channel], channel);

	}

}
