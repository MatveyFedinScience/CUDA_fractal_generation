#ifndef LAURENTSERIES_H
#define LAURENTSERIES_H

#include <map>
#include <complex>
#include <vector>
#include <iostream>

class LaurentSeries {

private:
    std::map<int, std::complex<double>> coeffs; 
public:
    LaurentSeries();
    void addTerm(int power, std::complex<double> coeff);
    void addTerm(int power, double coeff);
    void print() const;
    LaurentSeries differentiate() const;
    std::complex<double> evaluate(std::complex<double> z) const;
    std::vector<std::complex<double>> findRoots() const;

};

#endif // LAURENTSERIES_H
