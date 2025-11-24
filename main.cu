#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Для strcmp
#include <cuda_runtime.h>
#include "LaurentSeries.h"
#include "config.h"
#include "types.h"
#include "kernels.h" // Подключаем CUDA ядра
#include "helpers.h" // Подключаем функции хоста

int main(int argc, char *argv[]) {

    if (argc - 1 > 0) {
        printf("You have %d args to unpack, but this program ignore it!\n", argc - 1);
    }

    const char *output_filename = "mandelbrot.ppm";
    printf("Выходной файл: %s\n\n", output_filename);

    Complex z_start = { 0.0f, 0.0f };

    float xmin = -SQUARE_SIDE/2, xmax = SQUARE_SIDE/2;
    float ymin = -SQUARE_SIDE/2, ymax = SQUARE_SIDE/2;

    Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE];
    Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS];

    printf("%d;%d\n", COEFFS_SIZE, NUM_ROOTS);

    init_random_complex_array(coeffs_3d, -1.0f, 1.0f, -1.0f, 1.0f);
    printf("coeffs\n");
    print_complex_array_coeffs(coeffs_3d);

    process_coeffs_array(coeffs_3d, zeroes_3d);
    printf("roots \n");
    print_complex_array_roots(zeroes_3d);

    unsigned char *h_image = (unsigned char*)malloc(WIDTH * HEIGHT * XFRAC * YFRAC * 3);

    unsigned char *d_image;

    Complex *d_coeffs_3d;
    Complex *d_zeroes_3d;

    cudaMalloc(&d_image, WIDTH * HEIGHT * XFRAC * YFRAC * 3);
    cudaMalloc(&d_coeffs_3d, XFRAC * YFRAC * COEFFS_SIZE * sizeof(Complex));
    cudaMalloc(&d_zeroes_3d, XFRAC * YFRAC * NUM_ROOTS * sizeof(Complex));

    cudaMemcpy(d_coeffs_3d, coeffs_3d, XFRAC * YFRAC * COEFFS_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeroes_3d, zeroes_3d, XFRAC * YFRAC * NUM_ROOTS * sizeof(Complex), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(( WIDTH * XFRAC + 15 ) / 16, ( HEIGHT * YFRAC + 15 ) / 16);

    FRACTAL_KERNEL_FUNCTION<<<gridSize, blockSize>>>(d_image, d_coeffs_3d, d_zeroes_3d,
                                                     xmin, xmax, ymin, ymax);

    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * XFRAC * YFRAC * 3, cudaMemcpyDeviceToHost);

    save_split_images("fractal", h_image, "output");
//    save_ppm(output_filename, h_image, WIDTH * XFRAC, HEIGHT * YFRAC);
    save_to_csv("output.csv", coeffs_3d, zeroes_3d);

    cudaFree(d_image);
    cudaFree(d_coeffs_3d);
    cudaFree(d_zeroes_3d);

    free(h_image);
//    free(coeffs_3d);
//    free(zeroes_3d);


    return 0;

}
