#include <random>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <cmath>
#include "LaurentSeries.h"
#include <iostream>

void save_ppm(const char *filename, unsigned char *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Ошибка открытия файла для записи");
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image, 1, width * height * 3, fp);
    fclose(fp);
}

void save_to_csv(const char* filename,
                 Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE],
                 Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS]) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Ошибка открытия файла");
        return;
    }
    // Заголовок CSV
    for (int i = 0; i < REGRESSION_PARAMS; i++) {
        if (i < 2 * NUM_ROOTS) {
            if (i % 2 == 0) {
                fprintf(fp, "zero_real_%d", i/2);
            } else {
                fprintf(fp, "zero_imag_%d", i/2);
            }
        } else {
            int idx = i - 2 * NUM_ROOTS;
            if (idx % 2 == 0) {
                fprintf(fp, "coeff_real_%d", idx/2);
            } else {
                fprintf(fp, "coeff_imag_%d", idx/2);
            }
        }
        if (i < REGRESSION_PARAMS - 1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");
    
// Данные
    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
            
            // Zeroes
            for (int k = 0; k < NUM_ROOTS; k++) {
                fprintf(fp, "%.6f,%.6f", zeroes_3d[i][j][k].real, zeroes_3d[i][j][k].imag);
                if (k < NUM_ROOTS - 1) fprintf(fp, ",");
            }
            
            if (NUM_ROOTS > 0 && COEFFS_SIZE > 0) fprintf(fp, ",");
            
            // Coefficients
            for (int k = 0; k < COEFFS_SIZE; k++) {
                fprintf(fp, "%.6f,%.6f", coeffs_3d[i][j][k].real, coeffs_3d[i][j][k].imag);
                if (k < COEFFS_SIZE - 1) fprintf(fp, ",");
            }
            
            fprintf(fp, "\n");
        }
    }
    
    fclose(fp);
    printf("Сохранено в %s\n", filename);
}




void save_split_images(const char* base_filename, unsigned char* big_image, const char* folder = "output") {


    char command[256];
    snprintf(command, sizeof(command), "mkdir -p %s", folder);
    system(command);

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
            char filename[256];

            snprintf(filename, sizeof(filename), "%s/%s_%d_%d.ppm", folder, base_filename, i, j);

            FILE* fp = fopen(filename, "wb");
            fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

            for (int y = 0; y < HEIGHT; y++) {
                int big_y = j * HEIGHT + y;
                int big_idx = (big_y * WIDTH * XFRAC + i * WIDTH) * 3;
                fwrite(big_image + big_idx, 1, WIDTH * 3, fp);
            }

            fclose(fp);
        }
    }
}


void print_help() {
    printf("Использование:\n");
    printf("  ./mandelbrot <start_real> <start_imag> [output.ppm]\n\n");
    printf("Параметры:\n");
    printf("  start_real    - действительная часть начальной точки z0\n");
    printf("  start_imag    - мнимая часть начальной точки z0\n");
    printf("  output.ppm    - имя выходного файла (по умолчанию: mandelbrot.ppm)\n\n");
    printf("Примеры:\n");
    printf("  1) Стандартное множество Мандельброта f(z) = z^2:\n");
    printf("     ./mandelbrot 0 0\n\n");
    printf("  2) f(z) = -0.3*z + 0.8*z^3:\n");
    printf("     ./mandelbrot 0 0\n\n");
    printf("  3) f(z) = z^2 + 0.3/z с начальной точкой (0.1, 0):\n");
    printf("     ./mandelbrot 0.1 0\n\n");
    printf("  4) f(z) = (0.5+0.2i)*z^2:\n");
    printf("     ./mandelbrot 0 0 my_fractal.ppm\n\n");
}



void print_complex_array_coeffs(Complex arr[XFRAC][YFRAC][COEFFS_SIZE]) {
    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
             printf("[%d][%d]: { ", i, j);
            for (int k = 0; k < COEFFS_SIZE; k++) {
                printf("(%.2f%+.2fi) ", arr[i][j][k].real, arr[i][j][k].imag);
            }
            printf("}\n");
        }
    }
}

void print_complex_array_roots(Complex arr[XFRAC][YFRAC][NUM_ROOTS]) {
    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
             printf("[%d][%d]: { ", i, j);
            for (int k = 0; k < NUM_ROOTS; k++) {
                printf("(%.2f%+.2fi) ", arr[i][j][k].real, arr[i][j][k].imag);
            }
            printf("}\n");
        }
    }
}



void init_random_complex_array(Complex arr[XFRAC][YFRAC][COEFFS_SIZE],
                              float real_min, float real_max,
                              float imag_min, float imag_max) {
    //only for init coeffs

    std::mt19937 generator(RANDOM_SEED);

    std::uniform_real_distribution<float> real_dist(real_min, real_max);
    std::uniform_real_distribution<float> imag_dist(imag_min, imag_max);

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
            for (int k = 0; k < COEFFS_SIZE; k++) {
                arr[i][j][k].real = real_dist(generator);
                arr[i][j][k].imag = imag_dist(generator);
            }
        }
    }
}

void init_nan_complex_array(Complex arr[XFRAC][YFRAC][NUM_ROOTS]) {
    //only for init roots DEPRECATED
    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
            for (int k = 0; k < NUM_ROOTS; k++) {
                arr[i][j][k].real = NAN;
                arr[i][j][k].imag = NAN;
            }
        }
    }
}


void process_coeffs_array(Complex arr[XFRAC][YFRAC][COEFFS_SIZE],
                          Complex arr_roots[XFRAC][YFRAC][NUM_ROOTS]) {

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {

            LaurentSeries series;

            // coeffs_3d[i][j][k] соответствует степени MIN_POWER + k
            for (int k = 0; k < COEFFS_SIZE; k++) {
                int power = MIN_POWER + k;
                Complex coeff = arr[i][j][k];

                if (coeff.real != 0.0f || coeff.imag != 0.0f) {
                    series.addTerm(power, coeff);
                }
            }

            LaurentSeries deriv = series.differentiate();

            auto roots_d = deriv.findRoots();
            for (size_t idx = 0; idx < NUM_ROOTS; idx++) {
                arr_roots[i][j][idx] = Complex(roots_d[idx]);
            }
        }
    }
}
