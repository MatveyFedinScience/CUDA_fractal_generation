#include <cuda_runtime.h>
#include "config.h"
#include "types.h"

__device__ Complex complex_mult(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

__device__ Complex complex_power(Complex z, int n) {

    Complex result = {1.0f, 0.0f};

    int abs_n = abs(n);

    for (int iteration = 0; iteration < abs_n; iteration++){

    result = complex_mult(result, z);

    }

    Complex final_result;

    final_result.real = result.real / (
                -( (int)(n > 0) - 1 ) * (result.real * result.real + result.imag * result.imag) +
                  (int)(n > 0) );
    final_result.imag = result.imag * ( 2*(int)(n > 0) - 1 ) / (
                                -( (int)(n > 0) - 1 ) * (result.real * result.real + result.imag * result.imag) +
                                  (int)(n > 0) );
    return final_result;
}

__device__ Complex apply_laurent_series(Complex z, Complex *coeffs, int min_power, int max_power) {
    Complex result = {0.0f, 0.0f};


    for (int k = min_power; k <= max_power; k++) {
        int idx = k - min_power;
        Complex z_pow = complex_power(z, k);
        Complex term = complex_mult(coeffs[idx], z_pow);
        result.real += term.real;
        result.imag += term.imag;
    }

    return result;
}


__global__ void mandelbrot_kernel_color(unsigned char *image, Complex *coeffs_3d, Complex *roots,
                                        float xmin, float xmax, float ymin, float ymax) {

        int px = blockIdx.x * blockDim.x + threadIdx.x;
        int py = blockIdx.y * blockDim.y + threadIdx.y;

        // РЕКОМЕНДАЦИЯ: Добавить проверку границ массива, чтобы избежать записи
        // за пределы памяти, если размер сетки не кратен размерам изображения.
        // if (px >= WIDTH * XFRAC || py >= HEIGHT * YFRAC) return;

        float x0 = xmin + (xmax - xmin) * (px % WIDTH)  / (WIDTH);
        float y0 = ymin + (ymax - ymin) * (py % HEIGHT) / (HEIGHT);

        int min_power = MIN_POWER;
        int max_power = MAX_POWER;

        int ipx = px / WIDTH;
        int ipy = py / HEIGHT;

        Complex z;
        Complex c = {x0, y0};
        int iter_color = 0;

        Complex *coeffs = &coeffs_3d[ipx * (YFRAC * COEFFS_SIZE) + ipy * COEFFS_SIZE];

        // Итерация: z_{n+1} = f(z_n) + c, где f(z) - ряд Лорана
        for (int iteroot = 0; iteroot < NUM_ROOTS; iteroot++){
            z = roots[ipx * (YFRAC * NUM_ROOTS) + ipy * NUM_ROOTS + iteroot];
            for (int iteration = 0; iteration < MAX_ITER; iteration++) {
                z = apply_laurent_series(z, coeffs,
                                         min_power, max_power);
                z.real += c.real;
                z.imag += c.imag;
                iter_color += (int)(z.real > -10*SQUARE_SIDE / 2) *
                              (int)(10*SQUARE_SIDE / 2 > z.real) *
                              (int)(z.imag > -10*SQUARE_SIDE / 2) *
                              (int)(10*SQUARE_SIDE / 2 > z.imag);

            }
        }
//        iter_color = iter_color / NUM_ROOTS; WARNING!!!
        iter_color = fminf(iter_color, MAX_ITER);


        int idx = (py * WIDTH * XFRAC + px) * 3;
        float t = (float)(MAX_ITER - iter_color) / MAX_ITER;  // Вертикальный градиент
        image[idx+0] = (unsigned char)(t * 255);
        image[idx+1] = (unsigned char)(t * 255);
        image[idx+2] = (unsigned char)(t * 255);

}



__global__ void julia_kernel_color(unsigned char *image, Complex *coeffs_3d, Complex *ptr_c,
                                        float xmin, float xmax, float ymin, float ymax) {

        int px = blockIdx.x * blockDim.x + threadIdx.x;
        int py = blockIdx.y * blockDim.y + threadIdx.y;
        //CHECK CPTR!!! C IS JUST ONE VALUE VARIABLE!!!!! JUST FOR CONFOG COMPILATION CAPABILITY!!!
        // РЕКОМЕНДАЦИЯ: Добавить проверку границ массива, чтобы избежать записи
        // за пределы памяти, если размер сетки не кратен размерам изображения.
        // if (px >= WIDTH * XFRAC || py >= HEIGHT * YFRAC) return;

        float x0 = xmin + (xmax - xmin) * (px % WIDTH)  / (WIDTH);
        float y0 = ymin + (ymax - ymin) * (py % HEIGHT) / (HEIGHT);

        int min_power = MIN_POWER;
        int max_power = MAX_POWER;

        int ipx = px / WIDTH;
        int ipy = py / HEIGHT;

        Complex z = {x0, y0};
        Complex c = *ptr_c;
        int iter_color = 0;

        Complex *coeffs = &coeffs_3d[ipx * (YFRAC * COEFFS_SIZE) + ipy * COEFFS_SIZE];

        // Итерация: z_{n+1} = f(z_n) + c, где f(z) - ряд Лорана
        for (int iteration = 0; iteration < MAX_ITER; iteration++) {
            z = apply_laurent_series(z, coeffs,
                                     min_power, max_power);
            z.real += c.real;
            z.imag += c.imag;
            iter_color += (int)(z.real > -SQUARE_SIDE / 2) *
                          (int)(SQUARE_SIDE / 2 > z.real) *
                          (int)(z.imag > -SQUARE_SIDE / 2) *
                          (int)(SQUARE_SIDE / 2 > z.imag);

        }

        int idx = (py * WIDTH * XFRAC + px) * 3;
        //int idx = (py * WIDTH + px) * 3;
        float t = (float)(MAX_ITER - iter_color) / MAX_ITER;  // Вертикальный градиент
        image[idx+0] = (unsigned char)(t * 255);
        image[idx+1] = (unsigned char)(t * 255);
        image[idx+2] = (unsigned char)(t * 255);

}
