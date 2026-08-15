/* host_shim.h — replaces stm32f4xx.h when building velvet_reverb.c for host
 *
 * Provides no-op memory barriers and a forward declaration of the .bss array
 * that backs the simulated SDRAM T2 ring (defined in host_main.c).
 *
 * VELVET_REVERB_HOST is also defined here (and on the compiler command line)
 * so velvet_reverb.{h,c} swap in host-friendly shims for DMA + QADD.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifndef VELVET_REVERB_HOST
#define VELVET_REVERB_HOST 1
#endif

#define __DMB()  ((void)0)
#define __DSB()  ((void)0)
