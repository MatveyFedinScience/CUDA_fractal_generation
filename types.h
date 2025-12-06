#ifndef TYPES_H
#define TYPES_H

#include <complex>
#include <cmath>

#ifndef __CUDACC__
    #define __host__
    #define __device__
#endif

struct Complex {
    float real;
    float imag;

    __host__ __device__
    Complex() : real(0.0f), imag(0.0f) {}

    __host__ __device__
    Complex(float r, float i) : real(r), imag(i) {}

    __host__
    Complex(const std::complex<double>& z) 
        : real(static_cast<float>(z.real())), 
          imag(static_cast<float>(z.imag())) 
    {
    }

    __host__
    operator std::complex<double>() const {
        return std::complex<double>(real, imag);
    }

    __host__ __device__
    bool is_nan() const {
        #ifdef __CUDA_ARCH__
            return isnan(real) || isnan(imag);
        #else
            return std::isnan(real) || std::isnan(imag);
        #endif
    }
};

#endif // TYPES_H
