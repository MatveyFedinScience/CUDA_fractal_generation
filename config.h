#ifndef CONFIG_H
#define CONFIG_H

#define WIDTH (256*2)
#define HEIGHT (256*2)
#define MAX_ITER 256
#define MAX_COEFFS 32

#define SQUARE_SIDE 3.0f
#define SQUARE_SIDE_CH2 6.0f // Not implement yet TODO
#define SQUARE_SIDE_CH3 12.0f // Not implement yet TODO

#define CONF_SIDES_VALUES { SQUARE_SIDE, SQUARE_SIDE_CH2, SQUARE_SIDE_CH3 }
#define CONF_SIDES_COUNT 3


#define XFRAC 8
#define YFRAC 8
#define COEFFS_SIZE 2
#define MIN_POWER 2
#define MAX_POWER (MIN_POWER + COEFFS_SIZE - 1)
#define NUM_ROOTS ((MIN_POWER > 0) ? (MAX_POWER - 1) : (COEFFS_SIZE - 2)) //for derive
#define RANDOM_SEED 1
#define FRACTAL_KERNEL_FUNCTION mandelbrot_kernel_color
//use kernel with appropraite args types!!!
//FRACTAL_KERNEL_FUNCTION(unsigned char*, Complex*, Complex*, float, float, float, float)
//Now mandelbrot_kernel_color available and julia_kernel_color still in progress
#define REGRESSION_PARAMS ( 2 * NUM_ROOTS + 2 * COEFFS_SIZE )

#endif
