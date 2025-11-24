#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include "config.h"
#include "types.h"


__device__ Complex complex_mult(Complex a, Complex b);

__device__ Complex complex_power(Complex z, int n);


__device__ Complex apply_laurent_series(Complex z, Complex *coeffs, int min_power, int max_power);


__global__ void mandelbrot_kernel_color(unsigned char *image, Complex *coeffs_3d, Complex *roots,
                                        float xmin, float xmax, float ymin, float ymax);

#endif // KERNELS_H
