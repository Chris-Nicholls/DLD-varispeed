/*
 * params.c - take raw input data, smooth/debounce it, shift/scale,
 * and convert to params and modes
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

#define QUANTIZE_TIMECV_CH1 1
#define QUANTIZE_TIMECV_CH2 0

#include "globals.h"
#include "adc.h"
#include "dig_pins.h"
#include "params.h"
#include "equal_pow_pan_padded.h"
#include "exp_1voct.h"
#include "timekeeper.h"
#include "looping_delay.h"
#include "velvet_reverb.h"
#include "log_taper_padded.h"

extern const float log_taper[4096];
extern const float exp_1voct[4096];

extern __IO uint16_t potadc_buffer[NUM_POT_ADCS];
extern __IO uint16_t cvadc_buffer[NUM_CV_ADCS];



extern volatile uint32_t divmult_time[NUM_CHAN];


extern uint8_t flag_ping_was_changed[NUM_CHAN];
extern uint8_t flag_inf_change[NUM_CHAN];
extern uint8_t flag_rev_change[NUM_CHAN];
extern uint8_t flag_pot_changed_revdown[NUM_POT_ADCS];
extern uint8_t flag_pot_changed_infdown[NUM_POT_ADCS];
extern uint8_t flag_pot_changed_pingdown[NUM_POT_ADCS];

extern uint8_t disable_mode_changes;

extern uint8_t doing_reverse_fade[NUM_CHAN];


float param[NUM_CHAN][NUM_PARAMS];
float global_param[NUM_GLOBAL_PARAMS];
uint8_t mode[NUM_CHAN][NUM_CHAN_MODES];
uint8_t global_mode[NUM_GLOBAL_MODES];

uint8_t flag_time_param_changed[2];

int32_t MIN_POT_ADC_CHANGE[NUM_POT_ADCS];
int32_t MIN_CV_ADC_CHANGE[NUM_CV_ADCS];

extern int16_t CV_CALIBRATION_OFFSET[NUM_CV_ADCS];

float POT_LPF_COEF[NUM_POT_ADCS];
float CV_LPF_COEF[NUM_CV_ADCS];

//Low Pass filtered adc values:
float smoothed_potadc[NUM_POT_ADCS];
float smoothed_cvadc[NUM_CV_ADCS];

//Integer-ized low pass filtered adc values:
int16_t i_smoothed_potadc[NUM_POT_ADCS];
int16_t i_smoothed_cvadc[NUM_CV_ADCS];

int16_t old_i_smoothed_cvadc[NUM_CV_ADCS];
int16_t old_i_smoothed_potadc[NUM_POT_ADCS];

float smoothed_rawcvadc[NUM_CV_ADCS];
int16_t i_smoothed_rawcvadc[NUM_CV_ADCS];

//Change in pot since last process_adc
int32_t pot_delta[NUM_POT_ADCS];
int32_t cv_delta[NUM_POT_ADCS];


float set_fade_increment(uint32_t samples)
{
	return ( 1.0/((samples / (codec_BUFF_LEN>>3)) + 1.0) );
}

void init_params(void)
{
	uint8_t channel=0;

	for (channel=0;channel<NUM_CHAN;channel++){
		param[channel][TIME] = 0.0;
		param[channel][LEVEL] = 0.7;
		param[channel][REGEN] = 0.5;
		param[channel][MIX_DRY] = 0.7;
		param[channel][MIX_WET] = 0.7;
		param[channel][TRACKING_COMP] = 1.02;
		param[channel][VARISPEED_INERTIA] = 0.00083f;  // ~100ms slew time
	}

}

//initializes modes that aren't read from flash ram
void init_modes(void)
{
	uint8_t channel=0;

	for (channel=0;channel<NUM_CHAN;channel++){
		mode[channel][INF] = INF_OFF;
		mode[channel][REV] = 0;
		mode[channel][TIMEMODE_POT] = MOD_READWRITE_TIME_Q;
		mode[channel][TIMEMODE_JACK] = MOD_READWRITE_TIME_Q;

		mode[channel][PING_LOCKED] = 0;

		mode[channel][CONTINUOUS_REVERSE] = 0;
	}
	global_mode[DCINPUT] = 0;
	global_mode[CALIBRATE] = 0;
	global_mode[SYSTEM_SETTINGS] = 0;
	global_mode[QUANTIZE_MODE_CHANGES] = 0;


}


float LowPassSmoothingFilter(float current_value, float new_value, float coef)
{
	return (current_value * coef) + (new_value * (1.0f-coef));
}

void init_LowPassCoefs(void)
{
	float t;
	uint8_t i;

	t=50.0;

	CV_LPF_COEF[TIME*2] = 1.0-(1.0/t);
	CV_LPF_COEF[TIME*2+1] = 1.0-(1.0/t);

	t=10.0;

	CV_LPF_COEF[LEVEL*2] = 1.0-(1.0/t);
	CV_LPF_COEF[LEVEL*2+1] = 1.0-(1.0/t);

	CV_LPF_COEF[REGEN*2] = 1.0-(1.0/t);
	CV_LPF_COEF[REGEN*2+1] = 1.0-(1.0/t);

	MIN_CV_ADC_CHANGE[TIME*2] = 30;
	MIN_CV_ADC_CHANGE[TIME*2+1] = 30;

	MIN_CV_ADC_CHANGE[LEVEL*2] = 20;
	MIN_CV_ADC_CHANGE[LEVEL*2+1] = 20;

	MIN_CV_ADC_CHANGE[REGEN*2] = 20;
	MIN_CV_ADC_CHANGE[REGEN*2+1] = 20;


	t=20.0; //50.0 = about 100ms to turn a knob fully

	POT_LPF_COEF[TIME_POT*2] = 1.0-(1.0/t);
	POT_LPF_COEF[TIME_POT*2+1] = 1.0-(1.0/t);

	POT_LPF_COEF[LEVEL_POT*2] = 1.0-(1.0/t);
	POT_LPF_COEF[LEVEL_POT*2+1] = 1.0-(1.0/t);

	POT_LPF_COEF[REGEN_POT*2] = 1.0-(1.0/t);
	POT_LPF_COEF[REGEN_POT*2+1] = 1.0-(1.0/t);

	POT_LPF_COEF[MIX_POT*2] = 1.0-(1.0/t);
	POT_LPF_COEF[MIX_POT*2+1] = 1.0-(1.0/t);

	MIN_POT_ADC_CHANGE[TIME_POT*2] = 60;
	MIN_POT_ADC_CHANGE[TIME_POT*2+1] = 60;

	MIN_POT_ADC_CHANGE[LEVEL_POT*2] = 60;
	MIN_POT_ADC_CHANGE[LEVEL_POT*2+1] = 60;

	MIN_POT_ADC_CHANGE[REGEN_POT*2] = 60;
	MIN_POT_ADC_CHANGE[REGEN_POT*2+1] = 60;

	MIN_POT_ADC_CHANGE[MIX_POT*2] = 60;
	MIN_POT_ADC_CHANGE[MIX_POT*2+1] = 60;



	for (i=0;i<NUM_POT_ADCS;i++)
	{
		smoothed_potadc[i]=0;
		old_i_smoothed_potadc[i]=0;
		i_smoothed_potadc[i]=0x7FFF;
		pot_delta[i]=0;
	}
	for (i=0;i<NUM_CV_ADCS;i++)
	{
		smoothed_cvadc[i]=0;
		smoothed_rawcvadc[i]=0;
		old_i_smoothed_cvadc[i]=0;
		i_smoothed_cvadc[i]=0x7FFF;
		i_smoothed_rawcvadc[i]=0x7FFF;
		cv_delta[i]=0;
	}
}


void process_adc(void)
{
	uint8_t i;
	int32_t t;

	uint8_t flag_pot_changed[NUM_POT_ADCS];

	static uint32_t track_moving_pot[NUM_POT_ADCS]={0,0,0,0,0,0,0,0};

	//
	// Run a LPF on the pots and CV jacks
	//
	for (i=0;i<NUM_POT_ADCS;i++)
	{
		flag_pot_changed[i]=0;

		smoothed_potadc[i] = LowPassSmoothingFilter(smoothed_potadc[i], (float)potadc_buffer[i], POT_LPF_COEF[i]);
		i_smoothed_potadc[i] = (int16_t)smoothed_potadc[i];

		t=i_smoothed_potadc[i] - old_i_smoothed_potadc[i];

		//if (track_moving_pot[i])
		//	track_moving_pot[i]--;

		if ((t>MIN_POT_ADC_CHANGE[i]) || (t<-MIN_POT_ADC_CHANGE[i]))
			track_moving_pot[i]=250;

		if (track_moving_pot[i])
		{
			track_moving_pot[i]--;

			flag_pot_changed[i]=1;

			pot_delta[i] = t;

			//Todo: We could clean up the use of flag_pot_changed_XXXdown, instead change the mode right here

			//REV + TIME (in INF mode) is change loop end ---> disallow toggling Reverse
			if (REV1BUT && (i==TIME1_POT && mode[0][INF]==INF_ON))
				flag_pot_changed_revdown[i]=1;

			else if (REV2BUT && (i==TIME2_POT && mode[1][INF]==INF_ON))
				flag_pot_changed_revdown[i]=1;

			//INF + TIME is UNQ mode, INF + REGEN (in INF mode) is windowing ---> disallow toggling INF
			if (INF1BUT && (i==TIME1_POT || (i==REGEN1_POT && mode[0][INF])==INF_ON))
				flag_pot_changed_infdown[i]=1;

			if (INF2BUT && (i==TIME2_POT || (i==REGEN2_POT && mode[1][INF])==INF_ON))
				flag_pot_changed_infdown[i]=1;

			//PING + TIME sets varispeed inertia ---> disallow ping
			if (PINGBUT && i==TIME1_POT)
				flag_pot_changed_pingdown[i]=1;

			if (PINGBUT && i==TIME2_POT)
				flag_pot_changed_pingdown[i]=1;

			old_i_smoothed_potadc[i] = i_smoothed_potadc[i];
		}
	}

	for (i=0;i<NUM_CHAN;i++)
	{
		//PING + TIME sets varispeed inertia
		// min = 1ms, 12 o'clock = ~250ms, max = 1000ms
		if (flag_pot_changed_pingdown[TIME_POT*2+i])
		{
			float pot_val = (float)i_smoothed_potadc[TIME_POT*2+i] / 4095.0f;
			// Map pot: time_ms = 1 + 999 * pot^2, slew = 0.0833 / time_ms
			float time_factor = 1.0f + 999.0f * pot_val * pot_val;
			param[i][VARISPEED_INERTIA] = 0.0833f / time_factor;
		}

		if (flag_pot_changed_infdown[TIME_POT*2+i])
		{
			mode[i][TIMEMODE_POT]=MOD_READWRITE_TIME_NOQ;
			mode[i][TIMEMODE_JACK] = MOD_READWRITE_TIME_NOQ;
		}
		else if (flag_pot_changed[TIME_POT*2+i])
		{
			mode[i][TIMEMODE_POT]=MOD_READWRITE_TIME_Q;
			if (!global_mode[AUTO_UNQ])
				mode[i][TIMEMODE_JACK] = MOD_READWRITE_TIME_Q;

		}

		//Audio rate divmult time: auto switch to Unquantized mode
		//Disabled by default
		if (global_mode[AUTO_UNQ]){
			if (divmult_time[i] < 1024) // 48000 / 1023 = 47Hz
				mode[i][TIMEMODE_JACK] = MOD_READWRITE_TIME_NOQ;
			else
				mode[i][TIMEMODE_JACK] = MOD_READWRITE_TIME_Q;
		}


		if (flag_pot_changed_infdown[REGEN_POT*2+i])
		{
			mode[i][WINDOWMODE_POT]=WINDOW;
		}
		else if (flag_pot_changed[REGEN_POT*2+i])
		{
			mode[i][WINDOWMODE_POT]=NO_WINDOW;
		}
	}


	for (i=0;i<NUM_CV_ADCS;i++)
	{
		smoothed_cvadc[i] = LowPassSmoothingFilter(smoothed_cvadc[i], (float)(cvadc_buffer[i]+CV_CALIBRATION_OFFSET[i]), CV_LPF_COEF[i]);
		i_smoothed_cvadc[i] = (int16_t)smoothed_cvadc[i];
		if (i_smoothed_cvadc[i] < 0) i_smoothed_cvadc[i] = 0;
		if (i_smoothed_cvadc[i] > 4095) i_smoothed_cvadc[i] = 4095;

		if (global_mode[CALIBRATE])
		{
			smoothed_rawcvadc[i] = LowPassSmoothingFilter(smoothed_rawcvadc[i], (float)(cvadc_buffer[i]), CV_LPF_COEF[i]);
			i_smoothed_rawcvadc[i] = (int16_t)smoothed_rawcvadc[i];
			if (i_smoothed_rawcvadc[i] < 0) i_smoothed_rawcvadc[i] = 0;
			if (i_smoothed_rawcvadc[i] > 4095) i_smoothed_rawcvadc[i] = 4095;
		}

		t=i_smoothed_cvadc[i] - old_i_smoothed_cvadc[i];
		if ((t>MIN_CV_ADC_CHANGE[i]) || (t<-MIN_CV_ADC_CHANGE[i]))
		{
			cv_delta[i] = t;
			old_i_smoothed_cvadc[i] = i_smoothed_cvadc[i];
		}
	}

}




void update_params(void)
{
	uint8_t channel;
	int32_t t;

	int32_t t_combined;
	float temp_f;

	float abs_amt;
	uint8_t subtract;

	float time_mult[2];
	//time_mult[channel]=0.0;


	global_mode[DCINPUT]=(DCINPUT_JUMPER > 0)?1:0;

	for (channel=0;channel<2;channel++)
	{

//
// ******* TIME **********
//

		if (mode[channel][TIMEMODE_POT]==MOD_READWRITE_TIME_NOQ)
		{
			if (mode[channel][TIMEMODE_JACK]==MOD_READWRITE_TIME_NOQ) //Pot and Jack are unquantized
			{

				if (old_i_smoothed_cvadc[TIME*2+channel] < 2018) //positive voltage on the Time CV jack
				{
					t_combined = (2048-old_i_smoothed_cvadc[TIME*2+channel]) * param[channel][TRACKING_COMP];
					time_mult[channel] = get_clk_div_exact(old_i_smoothed_potadc[TIME_POT*2+channel])  / exp_1voct[t_combined];
				}
				else if (old_i_smoothed_cvadc[TIME*2+channel] > 2078)
				{
					t_combined = old_i_smoothed_potadc[TIME_POT*2+channel] + (old_i_smoothed_cvadc[TIME*2+channel]-2048);
					time_mult[channel] = get_clk_div_exact(t_combined);
				}
				else
				{
					time_mult[channel] = get_clk_div_exact(old_i_smoothed_potadc[TIME_POT*2+channel]);

				}

			}

			else //Pot is unquantized, Jack is quantized NOT WELL TESTED! BEWARE!
			{

				time_mult[channel] = get_clk_div_exact(i_smoothed_potadc[TIME_POT*2+channel]);
				if (i_smoothed_cvadc[TIME*2+channel] >= 2048)
					time_mult[channel] *= get_clk_div_nominal(i_smoothed_cvadc[TIME*2+channel]-2048);
				else
					time_mult[channel] = time_mult[channel] / get_clk_div_nominal(2048 - i_smoothed_cvadc[TIME*2+channel]);

			}
		}
		else
		{

			if (mode[channel][TIMEMODE_JACK]==MOD_READWRITE_TIME_Q) //Pot and Jack are quantized
			{

				if (i_smoothed_cvadc[TIME_POT*2+channel] <= 2048)
					time_mult[channel] = get_clk_div_nominal(old_i_smoothed_potadc[TIME_POT*2+channel]) * get_clk_div_nominal(2048 - i_smoothed_cvadc[TIME_POT*2+channel]);
				else
					time_mult[channel] = get_clk_div_nominal(old_i_smoothed_potadc[TIME_POT*2+channel]) / get_clk_div_nominal(i_smoothed_cvadc[TIME_POT*2+channel] - 2048);

			}

			else //Pot is quantized, Jack is unquantized  NOT WELL TESTED! BEWARE!
			{

				if (i_smoothed_cvadc[TIME*2+channel] <= 2048) //positive voltage on the Time CV jack
				{
					t_combined = (2048-old_i_smoothed_cvadc[TIME*2+channel]) * param[channel][TRACKING_COMP];
					time_mult[channel] = get_clk_div_nominal(i_smoothed_potadc[TIME_POT*2+channel])  / exp_1voct[t_combined];
				}
				else
				{
					time_mult[channel] = get_clk_div_nominal(i_smoothed_potadc[TIME_POT*2+channel]);
					time_mult[channel] *= get_clk_div_exact(old_i_smoothed_cvadc[TIME*2+channel]-2048);
				}


			}
		}

		time_mult[channel] = adjust_time_by_switch(time_mult[channel], channel);

		if (time_mult[channel] != param[channel][TIME])
		{
			flag_time_param_changed[channel] = 1;
			param[channel][TIME] = time_mult[channel];
		}



		if (mode[channel][INF]!=INF_ON)
		{
// *******  INF OFF **********
// ******* LEVEL **********

			if (mode[channel][LEVELCV_IS_MIX]==0)
			{
				t_combined = i_smoothed_potadc[LEVEL_POT*2+channel] + i_smoothed_cvadc[LEVEL*2+channel];
				asm("usat %[dst], #12, %[src]" : [dst] "=r" (t_combined) : [src] "r" (t_combined));
			}
			else
			{
				t_combined = i_smoothed_potadc[LEVEL_POT*2+channel];
			}

			/* Only channel 0 writes param[][LEVEL] from pot/CV. Channel 1's
			 * LEVEL is slaved from channel 0 at the end of update_params.
			 * If channel 1 also wrote here, the window between this write
			 * and the slaving would let the audio ISR read channel 1's
			 * value as the RIGHT LEVEL pot's value (which is repurposed as
			 * the density macro) — causing input audio to leak through
			 * mainin_atten when density > 0. */
			if (channel == 0) {
				if (global_mode[LOG_DELAY_FEED])
					param[channel][LEVEL] = log_taper[t_combined];

				else {
					if (t_combined<30.0)
						param[channel][LEVEL] = 0.0;

					else if (t_combined<4066.0)
						param[channel][LEVEL] = (t_combined - 30.0)/4066.0;

					else
						param[channel][LEVEL] = 1.0;
				}
			}




// ******* REGEN **********


			if (i_smoothed_potadc[REGEN_POT*2+channel]<3500.0)
				temp_f = i_smoothed_potadc[REGEN_POT*2+channel] / 3500.0;

			else if (i_smoothed_potadc[REGEN_POT*2+channel]<=4000.0)
				temp_f=1.0;

			else
				temp_f=(i_smoothed_potadc[REGEN_POT*2+channel]-3050)/950.0; // (4095-3050)/950 = 110% regeneration... (4000-3050)/950 = 100%

			if (i_smoothed_cvadc[REGEN*2+channel] > 30)
				temp_f = temp_f + (i_smoothed_cvadc[REGEN*2+channel] / 4096.0);

			else if (i_smoothed_cvadc[REGEN*2+channel] > 4080)
				temp_f = temp_f + 1.0;

			if (temp_f > 1.1)
				temp_f = 1.1;

			else if ((temp_f<1.003) && (temp_f>0.997))
				temp_f = 1.0;

			/* Same race-avoidance reasoning as LEVEL above. */
			if (channel == 0)
				param[channel][REGEN] = temp_f;


		}
		else
		{

// *******  INF ON **********

			if (channel == 0) {
				param[channel][LEVEL]=0.0;
				param[channel][REGEN]=1.0;
			}

// ********* WINDOW *********
			//
			// If REGEN was wiggled while INF is held down, then scroll the loop
			//
			t = pot_delta[REGEN_POT*2+channel] + cv_delta[REGEN_POT*2+channel];

			if (mode[channel][WINDOWMODE_POT]==WINDOW && (t != 0))
			{
				if (t < 0)
				{
					abs_amt = t / -4096.0;
					subtract = 1;
				} else
				{
					abs_amt = t / 4096.0;
					subtract = 0;
				}

				scroll_loop(channel, abs_amt, subtract);

				pot_delta[REGEN_POT*2+channel]=0;
				cv_delta[REGEN*2+channel]=0;

			}

		}

// ******* MIX **********

		//
		// MIX uses an equal power panning lookup table
		// Each MIX pot sets two parameters: wet and dry
		//

		if (mode[channel][LEVELCV_IS_MIX])
		{
			t_combined = i_smoothed_potadc[MIX_POT*2+channel] + i_smoothed_cvadc[LEVEL*2+channel];
			asm("usat %[dst], #12, %[src]" : [dst] "=r" (t_combined) : [src] "r" (t_combined));
		}
		else
		{
			t_combined = i_smoothed_potadc[MIX_POT*2+channel];
		}

		/* Same race-avoidance: param[1][MIX_DRY/MIX_WET] is slaved from
		 * channel 0 below, not from the right MIX pot. The right MIX pot's
		 * wet value is captured into reverb_send separately. */
		if (channel == 0) {
			param[channel][MIX_DRY]=epp_lut[t_combined];
			param[channel][MIX_WET]=epp_lut[4095 - t_combined];
		}

	}

	/* ===================================================================
	 * Reverb macro wiring + mix split
	 *
	 * Right channel's LEVEL / REGEN pots+CVs are repurposed for Density /
	 * Decay macros. Left channel's LEVEL / REGEN mirror to the right so
	 * both delay channels are slaved to one set of knobs.
	 *
	 * Mix split (per user request):
	 *   Left  MIX → delay internal dry/wet (both delay channels track this)
	 *   Right MIX → reverb SEND amount (scales the input feeding the reverb;
	 *               the reverb output is always mixed in unscaled).
	 *               Equal-power LUT — same curve as before, but now applied
	 *               to the input side of the reverb rather than crossfading
	 *               its output.
	 *
	 * Capture reverb_send from param[1][MIX_WET] BEFORE we slave param[1]'s
	 * MIX_DRY/WET to param[0].
	 * =================================================================== */
	{
		int32_t t;

		/* Right LEVEL → Density macro */
		t = (int32_t)i_smoothed_potadc[LEVEL_POT*2+1] + (int32_t)i_smoothed_cvadc[LEVEL*2+1];
		if (t < 0) t = 0;
		if (t > 4095) t = 4095;
		velvet_reverb_apply_density_macro((float)t * (1.0f / 4095.0f));

		/* Right REGEN → Decay macro */
		t = (int32_t)i_smoothed_potadc[REGEN_POT*2+1] + (int32_t)i_smoothed_cvadc[REGEN*2+1];
		if (t < 0) t = 0;
		if (t > 4095) t = 4095;
		velvet_reverb_apply_decay_macro((float)t * (1.0f / 4095.0f));

		/* Right MIX → reverb send. Computed directly from the right MIX
		 * pot (+ LEVEL CV if LEVELCV_IS_MIX is set, mirroring the standard
		 * MIX logic above). NOT via param[1][MIX_WET] which is now slaved
		 * from channel 0, not from the right pot. */
		if (mode[1][LEVELCV_IS_MIX]) {
			t = (int32_t)i_smoothed_potadc[MIX_POT*2+1] + (int32_t)i_smoothed_cvadc[LEVEL*2+1];
			asm("usat %[dst], #12, %[src]" : [dst] "=r" (t) : [src] "r" (t));
		} else {
			t = (int32_t)i_smoothed_potadc[MIX_POT*2+1];
			if (t < 0) t = 0;
			if (t > 4095) t = 4095;
		}
		reverb_send = epp_lut[4095 - t];

		/* Last 10% of the right MIX pot fades the dry+delay signal out so
		 * that at max the output is 100% wet reverb. Below 90% the dry path
		 * is unaffected; from 90%→100% it ramps linearly to 0. */
		{
			const int32_t fade_start = (4095 * 9) / 10;   /* 3685 */
			if (t <= fade_start) {
				reverb_dry_gain = 1.0f;
			} else {
				reverb_dry_gain = (float)(4095 - t) * (1.0f / (float)(4095 - fade_start));
			}
		}

		/* Left LEVEL / REGEN / MIX drive both delay channels. With the
		 * loop body's writes to param[1] guarded out, this slaving is the
		 * ONLY place param[1][LEVEL/REGEN/MIX_DRY/MIX_WET] gets set — no
		 * intermediate window where it holds right-pot values that would
		 * leak through the audio ISR. */
		param[1][LEVEL]   = param[0][LEVEL];
		param[1][REGEN]   = param[0][REGEN];
		param[1][MIX_DRY] = param[0][MIX_DRY];
		param[1][MIX_WET] = param[0][MIX_WET];
	}
}


//
// Handle all flags to change modes: INF, REV, and ping or div/mult time
//
void process_mode_flags(uint8_t channel)
{
	if (!disable_mode_changes)
	{
		if (flag_inf_change[channel])
		{
			change_inf_mode(channel);
			//mode[channel][CONTINUOUS_REVERSE] = 0;
		}

		if (flag_rev_change[channel])
		{
			//mode[channel][CONTINUOUS_REVERSE] = 0;

			if (!doing_reverse_fade[channel])
			{
				flag_rev_change[channel]=0;

				mode[channel][REV] = 1- mode[channel][REV];

				if (mode[channel][INF]==INF_ON || mode[channel][INF]==INF_TRANSITIONING_OFF || mode[channel][INF]==INF_TRANSITIONING_ON)
					reverse_loop(channel);

				else
					swap_read_write(channel);
			}
		}
	}

	if (flag_time_param_changed[channel] || flag_ping_was_changed[channel])
	{
		flag_time_param_changed[channel]=0;
		flag_ping_was_changed[channel]=0;
		set_divmult_time(channel);
	}

}


void process_ping_changed(uint8_t channel)
{

	if (flag_ping_was_changed[channel])
	{
		flag_time_param_changed[channel]=0;
		flag_ping_was_changed[channel]=0;
		set_divmult_time(channel);
	}

}


uint8_t get_switch_val(uint8_t channel)
{

	if (channel==0)
		return (TIMESW_CH1);
	else
		return (TIMESW_CH2);
}


// Adjust TIME by the time switch position
float adjust_time_by_switch(float val, uint8_t channel){
	uint8_t switch_val = get_switch_val(channel);

	if (switch_val==0b10) return(val + 16.0); //switch up: 17-32
	if (switch_val==0b01) return(val * 0.125); //switch down: eighth notes
	return val;
}


float get_clk_div_nominal(uint16_t adc_val)
{
	if (adc_val<=40) //was 150
		return(P_1); //1
	else if (adc_val<=176) //was 310
		return(P_2); //1.5
	else if (adc_val<=471)
		return(P_3); //2
	else if (adc_val<=780)
		return(P_4); //3
	else if (adc_val<=1076)
		return(P_5); //4
	else if (adc_val<=1368)
		return(P_6); //5
	else if (adc_val<=1664)
		return(P_7); //6
	else if (adc_val<=1925)
		return(P_8); //7
	else if (adc_val<=2179) // Center
		return(P_9); //8
	else if (adc_val<=2448)
		return(P_10); //9
	else if (adc_val<=2714)
		return(P_11); //10
	else if (adc_val<=2991)
		return(P_12); //11
	else if (adc_val<=3276)
		return(P_13); //12
	else if (adc_val<=3586)
		return(P_14); //13
	else if (adc_val<=3879)
		return(P_15); //14
	else if (adc_val<=4046)
		return(P_16); //15
	else
		return(P_17); //16
}

float get_clk_div_exact(uint16_t adc_val)
{
	float t, b, tval, bval;

	if (adc_val<=25)
		return(P_1);
	else if (adc_val<=108)
	{t=108;b=25;tval=P_2;bval=P_1;}
	else if (adc_val<=323)
	{t=323;b=108;tval=P_3;bval=P_2;}
	else if (adc_val<=625)
	{t=625;b=323;tval=P_4;bval=P_3;}
	else if (adc_val<=928)
	{t=928;b=625;tval=P_5;bval=P_4;}
	else if (adc_val<=1222)
	{t=1222;b=925;tval=P_6;bval=P_5;}
	else if (adc_val<=1516)
	{t=1516;b=1222;tval=P_7;bval=P_6;}
	else if (adc_val<=1794)
	{t=1794;b=1516;tval=P_8;bval=P_7;}
	else if (adc_val<=2052)
	{t=2052;b=1794;tval=P_9;bval=P_8;}
	else if (adc_val<=2581)
	{t=2581;b=2052;tval=P_10;bval=P_9;}
	else if (adc_val<=2852)
	{t=2852;b=2581;tval=P_11;bval=P_10;}
	else if (adc_val<=3133)
	{t=3133;b=2852;tval=P_12;bval=P_11;}
	else if (adc_val<=3431)
	{t=3431;b=3133;tval=P_13;bval=P_12;}
	else if (adc_val<=3732)
	{t=3732;b=3431;tval=P_14;bval=P_13;}
	else if (adc_val<=3962)
	{t=3962;b=3732;tval=P_15;bval=P_14;}
	else if (adc_val<=4062)
	{t=4062;b=3962;tval=P_16;bval=P_15;}
	else
		return(P_17);

	return( ((t-adc_val)/(t-b))*bval + ((adc_val-b)/(t-b))*tval );
}
