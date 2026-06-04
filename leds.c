/*
 * leds.c - driver for all LEDs (buttons and discrete)
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


#include <dig_pins.h>
#include "globals.h"
#include "leds.h"
#include "timekeeper.h"
#include "params.h"
#include "calibration.h"
#include "system_settings.h"


extern volatile uint32_t ping_ledbut_tmr;
extern volatile uint32_t loopled_tmr[2];
extern volatile uint32_t divmult_time[2];
extern volatile uint32_t ping_time;
extern uint8_t mode[NUM_CHAN][NUM_CHAN_MODES];
extern uint8_t global_mode[NUM_GLOBAL_MODES];
extern float global_param[NUM_GLOBAL_PARAMS];

extern uint32_t flag_acknowlegde_qcm;

//extern uint32_t flash_firmware_version;

uint8_t loop_led_state[NUM_CHAN]={0,0};


void update_ping_ledbut(void)
{
	if (global_mode[CALIBRATE])
		LED_PINGBUT_OFF;
	else if (global_mode[SYSTEM_SETTINGS])
	{
		//LED_PINGBUT_OFF;
	}
	else if (flag_acknowlegde_qcm)
	{
		flag_acknowlegde_qcm--;
		if (
				(flag_acknowlegde_qcm & (1<<8))
				|| (!global_mode[QUANTIZE_MODE_CHANGES] && (flag_acknowlegde_qcm & (1<<6)))
				)
		{
			LED_PINGBUT_ON;
			LED_REV1_ON;
			LED_REV2_ON;
		}else{
			LED_PINGBUT_OFF;
			LED_REV1_OFF;
			LED_REV2_OFF;
		}

	}
	else
	{
		if (ping_ledbut_tmr>=ping_time)
		{
			LED_PINGBUT_ON;
			reset_ping_ledbut_tmr();
		}
		else if (ping_ledbut_tmr >= (ping_time>>1))
		{
			LED_PINGBUT_OFF;
		}
	}
}

/*
  			|| (mode[channel][WINDOWMODE_POT]==WINDOW && (flicker_ctr & 0x3FFFF) <= 0x5000)
			|| (mode[channel][TIMEMODE_POT]==MOD_READWRITE_TIME_NOQ && (flicker_ctr & 0xFFFF) > 0xC000)

 */

void chase_all_lights(uint32_t delaytime)
{

		LED_LOOP1_ON;
		delay_ms(delaytime);
		LED_LOOP1_OFF;
		delay_ms(delaytime);

		LED_PINGBUT_ON;
		delay_ms(delaytime);
		LED_PINGBUT_OFF;
		delay_ms(delaytime);

		LED_LOOP2_ON;
		delay_ms(delaytime);
		LED_LOOP2_OFF;
		delay_ms(delaytime);

		LED_REV1_ON;
		delay_ms(delaytime);
		LED_REV1_OFF;
		delay_ms(delaytime);

		LED_INF1_ON;
		delay_ms(delaytime);
		LED_INF1_OFF;
		delay_ms(delaytime);

		LED_INF2_ON;
		delay_ms(delaytime);
		LED_INF2_OFF;
		delay_ms(delaytime);

		LED_REV2_ON;
		delay_ms(delaytime);
		LED_REV2_OFF;
		delay_ms(delaytime);

		//infinite loop to force user to reset

}


void blink_all_lights(uint32_t delaytime)
{

		LED_LOOP1_ON;
		LED_PINGBUT_ON;
		LED_LOOP2_ON;
		LED_REV1_ON;
		LED_INF1_ON;
		LED_INF2_ON;
		LED_REV2_ON;
		delay_ms(delaytime);

		LED_LOOP1_OFF;
		LED_PINGBUT_OFF;
		LED_LOOP2_OFF;
		LED_REV1_OFF;
		LED_INF1_OFF;
		LED_INF2_OFF;
		LED_REV2_OFF;
		delay_ms(delaytime);

}




void update_channel_leds(void)
{
	uint8_t channel;

	for (channel=0;channel<NUM_CHAN;channel++)
	{
		//if (!mode[channel][CONTINUOUS_REVERSE])
		//{
			if (loopled_tmr[channel] >= divmult_time[channel] && (mode[channel][INF]==INF_OFF))
			{
				reset_loopled_tmr(channel);
			}

			else if (loopled_tmr[channel] >= (divmult_time[channel]>>1))
			{

				set_loop_led(channel, 0);

				if (channel==0) {
					CLKOUT1_OFF;
				} else {
					CLKOUT2_OFF;
				}
			}

			else if (mode[channel][LOOP_CLOCK_GATETRIG] == TRIG_MODE && loopled_tmr[channel] >= TRIG_TIME)
			{
				if (channel==0)
					CLKOUT1_OFF;
				 else
					CLKOUT2_OFF;

			}
		//}
	}
}

void update_INF_REV_ledbut(uint8_t channel)
{
	static uint32_t flicker_ctr=0;
	uint8_t t;

	flicker_ctr-=(1<<25);

	if (global_mode[CALIBRATE])
	{
		update_calibration_button_leds();
	}
	else if (global_mode[SYSTEM_SETTINGS])
	{
		update_system_settings_button_leds();
	}
	else
	{

		//let the ping button function handle the rev lights blinking in ack_qcm state
		if (!flag_acknowlegde_qcm)
		{

			//create a flicker by inverting the led state
			t = mode[channel][CONTINUOUS_REVERSE] && (flicker_ctr<(1<<23));

			if (mode[channel][REV] == t)
			{
				if (channel==0)	LED_REV1_OFF;
				else 			LED_REV2_OFF;
			} else
			{
				if (channel==0)	LED_REV1_ON;
				else			LED_REV2_ON;
			}

		}
		//create a flicker by inverting the state
		t = mode[channel][PING_LOCKED] && (flicker_ctr<(1<<28));
		if ((mode[channel][INF]!=INF_ON && mode[channel][INF]!=INF_TRANSITIONING_ON) == t)
		{
			if (channel==0)	LED_INF1_ON;
			else 			LED_INF2_ON;
		}
		else
		{
			if (channel==0) LED_INF1_OFF;
			else			LED_INF2_OFF;
		}
	}

}

void init_LED_PWM_IRQ(void)
{
	TIM_TimeBaseInitTypeDef tim;
	NVIC_InitTypeDef nvic;

	RCC_APB1PeriphClockCmd(LED_TIM_RCC, ENABLE);

	nvic.NVIC_IRQChannel = LED_TIM_IRQn;
	nvic.NVIC_IRQChannelPreemptionPriority = 0;
	nvic.NVIC_IRQChannelSubPriority = 2;
	nvic.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&nvic);

	TIM_TimeBaseStructInit(&tim);
	/* APB1 timer clock = HCLK = 180 MHz (APB1 div 4 → 45, ×2 since div != 1
	 * → 90 MHz; wait, actually APB1 timer is APB1×2 when PCLK1 prescaler != 1,
	 * giving 90 MHz). Period 4688 → 4688/90e6 ≈ 52 µs ≈ 19.2 kHz. Same rate
	 * as 168 MHz config (period 4375 / 84 MHz = 19.2 kHz). */
	tim.TIM_Period = 4688;
	tim.TIM_Prescaler = 0;
	tim.TIM_ClockDivision = 0;
	tim.TIM_CounterMode = TIM_CounterMode_Up;

	TIM_TimeBaseInit(LED_TIM, &tim);
	TIM_ITConfig(LED_TIM, TIM_IT_Update, ENABLE);
	TIM_Cmd(LED_TIM, ENABLE);
}

/* 19.2 kHz PWM driving both loop LEDs. The 32-step counter gives a ~600 Hz
 * PWM-cycle frequency, well above the click rate (4–6 Hz at loop tempo).
 * Restored after observation that removing this caused channel-1 audio
 * clicks at the loop rate — apparently the slow GPIO edges at the loop
 * rate were coupling into codec B's analog output. */
void LED_PWM_IRQHandler(void)
{
	static uint32_t loop_led_PWM_ctr = 0;
	if (TIM_GetITStatus(TIM2, TIM_IT_Update) != RESET) {
		if (loop_led_state[0] && (loop_led_PWM_ctr < global_param[LOOP_LED_BRIGHTNESS]))
			LED_LOOP1_ON;
		else
			LED_LOOP1_OFF;

		if (loop_led_state[1] && (loop_led_PWM_ctr < global_param[LOOP_LED_BRIGHTNESS]))
			LED_LOOP2_ON;
		else
			LED_LOOP2_OFF;

		if (loop_led_PWM_ctr++ > 32)
			loop_led_PWM_ctr = 0;

		TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
	}
}
