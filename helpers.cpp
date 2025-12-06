#include <cstring>
#include <random>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "helpers.h"
#include <cmath>
#include "LaurentSeries.h"
#include <iostream>
#include <zlib.h>


void generate_names(Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE], char names[XFRAC][YFRAC][64]) {

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {

            unsigned char data[COEFFS_SIZE * sizeof(Complex)];
            memcpy(data, coeffs_3d[i][j], COEFFS_SIZE * sizeof(Complex));

            uLong crc = crc32(0L, Z_NULL, 0);
            crc = crc32(crc, data, COEFFS_SIZE * sizeof(Complex));

            snprintf(names[i][j], 64, "%08lx", crc);
        }
    }
}


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
                 Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS],
                 char    names[XFRAC][YFRAC][64]) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Ошибка открытия файла");
        return;
    }

    fprintf(fp, "filename,");
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

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {

            //Filename
            fprintf(fp, "%s,", names[i][j]);

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


void save_split_images(char names[XFRAC][YFRAC][64], unsigned char* big_image, const char* folder = "output") {


    char command[256];
    snprintf(command, sizeof(command), "mkdir -p %s", folder);
    system(command);

    for (int i = 0; i < XFRAC; i++) {
        for (int j = 0; j < YFRAC; j++) {
            char filename[256];

            snprintf(filename, sizeof(filename), "%s/%s.ppm", folder, names[i][j]);

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
    printf("Usage: ./mandelbrot [input.csv] [-c compare.csv]\n\n");
    printf("Options:\n");
    printf("  -h, --help           Show this help\n");
    printf("  -c, --compare FILE   Compare mode: ./mandelbrot file1.csv -c file2.csv\n");
    printf("                       Outputs difference metric [0.0-1.0]\n\n");
    printf("Examples:\n");
    printf("  ./mandelbrot                       # Random fractals\n");
    printf("  ./mandelbrot data.csv              # Generate from CSV\n");
    printf("  ./mandelbrot a.csv -c b.csv        # Compare two CSVs\n");
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


void init_from_file_complex_array(const char* filename,
                 Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE],
                 Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS],
                 char    names[XFRAC][YFRAC][64]) {
/*

IT WORKS STABLE ONLY FOR SMALL DATASET! TODO FIXME
IF FUNCTION DOESN'T WORK CHECK YOUR TOTAL SHAPE IS LESS THEN 128 INSTANCES

IN FILE ZEROES IS FIRST AND COEFFS IS LAST AS IN A save_to_csv
WARNING: IF in filename we have lower than XFRAC*YFRAC rows it generate zerocoeffs fractals
with name default

In general, we should to read files with ability to be XFRAC*YFRAC len

*/
    printf("WARNING: THIS FUNCTION DOESN'T WORK WITH MORE THEN 128 INSTANCES! YOU USE %d.\n", XFRAC*YFRAC);
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Warning: Cannot open file %s. Using defaults.\n", filename);
        return;
    }
    char line[8192];

    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return;
    }
    int row = 0;
    while (row < XFRAC * YFRAC && fgets(line, sizeof(line), file)) {
        int i = row / YFRAC;
        int j = row % YFRAC;

//        printf("DEBUG: Processing row %d/%d\n", row, XFRAC * YFRAC);

        char* ptr = line;
        
        char* comma = strchr(ptr, ',');
        if (comma) {
            size_t len = comma - ptr;
            if (len > 63) len = 63;
            strncpy(names[i][j], ptr, len);
            names[i][j][len] = '\0';
            ptr = comma + 1;
        } else {
            strcpy(names[i][j], "default");
            row++;
            continue;
        }

        for (int k = 0; k < NUM_ROOTS && ptr; k++) {
            float real, imag;
            int consumed = 0;
            if (sscanf(ptr, "%f,%f%n", &real, &imag, &consumed) == 2) {
                zeroes_3d[i][j][k].real = real;
                zeroes_3d[i][j][k].imag = imag;
                ptr += consumed;
                if (*ptr == ',') ptr++;
            } else {
                break;
            }
        }

        for (int k = 0; k < COEFFS_SIZE && ptr; k++) {
            float real, imag;
            int consumed = 0;
            if (sscanf(ptr, "%f,%f%n", &real, &imag, &consumed) == 2) {
                coeffs_3d[i][j][k].real = real;
                coeffs_3d[i][j][k].imag = imag;
                ptr += consumed;
                if (*ptr == ',') ptr++;
            } else {
                break;
            }
        }

        row++;
    }
    fclose(file);
    for (; row < XFRAC * YFRAC; row++) {
        int i = row / YFRAC;
        int j = row % YFRAC;
        strcpy(names[i][j], "default");
        memset(zeroes_3d[i][j], 0, NUM_ROOTS * sizeof(Complex));
        memset(coeffs_3d[i][j], 0, COEFFS_SIZE * sizeof(Complex));
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
