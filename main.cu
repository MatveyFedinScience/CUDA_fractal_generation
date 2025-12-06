#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "LaurentSeries.h"
#include "config.h"
#include "types.h"
#include "kernels.h"
#include "helpers.h"


int main(int argc, char *argv[]) {
    // TODO flags like rewrite zeroes for coeffs and rewrite coeffs for zeroes

    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_help();
        return 0;
    }

    bool compare_mode = false;
    const char *input_filename1 = nullptr;
    const char *input_filename2 = nullptr;

    if (argc >= 4 && (strcmp(argv[2], "-c") == 0 || strcmp(argv[2], "--compare") == 0)) {
        compare_mode = true;
        input_filename1 = argv[1];
        input_filename2 = argv[3];
        printf("Compare mode: %s vs %s\n", input_filename1, input_filename2);

        FILE* f1 = fopen(input_filename1, "r");
        FILE* f2 = fopen(input_filename2, "r");
        if (!f1 || !f2) {
            printf("Error: Cannot open input files for comparison\n");
            if (f1) fclose(f1);
            if (f2) fclose(f2);
            return 1;
        }
        fclose(f1);
        fclose(f2);
    } else if (argc > 1) {
        input_filename1 = argv[1];
        printf("Using input file: %s\n", input_filename1);
        FILE* test_file = fopen(input_filename1, "r");
        if (!test_file) {
            printf("Warning: Cannot open input file %s, using random initialization\n", input_filename1);
            input_filename1 = nullptr;
        } else {
            fclose(test_file);
        }
    } else {
        printf("No input file provided, using random initialization\n");
    }

    float xmin = -SQUARE_SIDE/2, xmax = SQUARE_SIDE/2;
    float ymin = -SQUARE_SIDE/2, ymax = SQUARE_SIDE/2;

    Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE];
    Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS];
    char names[XFRAC][YFRAC][64];

    // Загрузка первой конфигурации
    if (input_filename1) {
        init_from_file_complex_array(input_filename1, coeffs_3d, zeroes_3d, names);
    } else {
        init_random_complex_array(coeffs_3d, -1.0f, 1.0f, -1.0f, 1.0f);
        generate_names(coeffs_3d, names);
        process_coeffs_array(coeffs_3d, zeroes_3d);
    }

    unsigned char *h_image1 = (unsigned char*)malloc(WIDTH * HEIGHT * XFRAC * YFRAC * 3);
    unsigned char *d_image;
    Complex *d_coeffs_3d, *d_zeroes_3d;

    cudaMalloc(&d_image, WIDTH * HEIGHT * XFRAC * YFRAC * 3);
    cudaMalloc(&d_coeffs_3d, XFRAC * YFRAC * COEFFS_SIZE * sizeof(Complex));
    cudaMalloc(&d_zeroes_3d, XFRAC * YFRAC * NUM_ROOTS * sizeof(Complex));

    cudaMemcpy(d_coeffs_3d, coeffs_3d, XFRAC * YFRAC * COEFFS_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeroes_3d, zeroes_3d, XFRAC * YFRAC * NUM_ROOTS * sizeof(Complex), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH * XFRAC + 15) / 16, (HEIGHT * YFRAC + 15) / 16);

    FRACTAL_KERNEL_FUNCTION<<<gridSize, blockSize>>>(d_image, d_coeffs_3d, d_zeroes_3d, xmin, xmax, ymin, ymax);
    cudaDeviceSynchronize();
    cudaMemcpy(h_image1, d_image, WIDTH * HEIGHT * XFRAC * YFRAC * 3, cudaMemcpyDeviceToHost);

    if (compare_mode) {
        Complex coeffs_3d2[XFRAC][YFRAC][COEFFS_SIZE];
        Complex zeroes_3d2[XFRAC][YFRAC][NUM_ROOTS];
        char names2[XFRAC][YFRAC][64];
        
        init_from_file_complex_array(input_filename2, coeffs_3d2, zeroes_3d2, names2);

        unsigned char *h_image2 = (unsigned char*)malloc(WIDTH * HEIGHT * XFRAC * YFRAC * 3);

        cudaMemcpy(d_coeffs_3d, coeffs_3d2, XFRAC * YFRAC * COEFFS_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zeroes_3d, zeroes_3d2, XFRAC * YFRAC * NUM_ROOTS * sizeof(Complex), cudaMemcpyHostToDevice);

        FRACTAL_KERNEL_FUNCTION<<<gridSize, blockSize>>>(d_image, d_coeffs_3d, d_zeroes_3d, xmin, xmax, ymin, ymax);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image2, d_image, WIDTH * HEIGHT * XFRAC * YFRAC * 3, cudaMemcpyDeviceToHost);

        size_t total_pixels = WIDTH * HEIGHT * XFRAC * YFRAC * 3;
        double diff_sum = 0.0;
        double unio_sum = total_pixels;

        for (size_t i = 0; i < total_pixels; i++) {
            float val1 = h_image1[i] / 255.0f;
            float val2 = h_image2[i] / 255.0f;
            diff_sum += fabs(val1 - val2);
        }

        for (size_t i = 0; i < total_pixels; i++) {
            float val1 = 1.0f - h_image1[i] / 255.0f;
            float val2 = 1.0f - h_image2[i] / 255.0f;
            unio_sum -= val1 * val2;
        }

        double metric = diff_sum / unio_sum;

        printf("\n===========================================\n");
        printf("Comparison Metric (unsimilarity): %.6f", metric);
        printf("\n===========================================\n");

        free(h_image2);
    } else {
        printf("\nCoefficients:\n");
        print_complex_array_coeffs(coeffs_3d);
        printf("\nRoots:\n");
        print_complex_array_roots(zeroes_3d);

        save_split_images(names, h_image1, "output");
        save_to_csv("output.csv", coeffs_3d, zeroes_3d, names);
    }

    cudaFree(d_image);
    cudaFree(d_coeffs_3d);
    cudaFree(d_zeroes_3d);
    free(h_image1);

    return 0;
}
