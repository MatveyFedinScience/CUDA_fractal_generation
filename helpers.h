#ifndef HELPERS_H
#define HELPERS_H

#include "config.h"
#include "types.h"

void save_ppm(const char *filename, unsigned char *image, int width, int height);
void save_to_csv(const char* filename, 
                 Complex coeffs_3d[XFRAC][YFRAC][COEFFS_SIZE], 
                 Complex zeroes_3d[XFRAC][YFRAC][NUM_ROOTS]);
void save_split_images(const char* base_filename, unsigned char* big_image, const char* folder);

void print_help();
void print_complex_array_coeffs(Complex arr[XFRAC][YFRAC][COEFFS_SIZE]);
void print_complex_array_roots(Complex arr[XFRAC][YFRAC][NUM_ROOTS]);

void init_random_complex_array(Complex arr[XFRAC][YFRAC][COEFFS_SIZE],
                              float real_min, float real_max,
                              float imag_min, float imag_max);
void init_nan_complex_array(Complex arr[XFRAC][YFRAC][NUM_ROOTS]);
void process_coeffs_array(Complex arr[XFRAC][YFRAC][COEFFS_SIZE],
                          Complex arr_roots[XFRAC][YFRAC][NUM_ROOTS]);

#endif // HELPERS_H
