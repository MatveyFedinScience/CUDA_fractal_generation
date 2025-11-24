#include "LaurentSeries.h"
#include <cmath>
#include <Eigen/Eigenvalues>

LaurentSeries::LaurentSeries() {}

// Добавить коэффициент a * z^power (комплексный)
void LaurentSeries::addTerm(int power, std::complex<double> coeff) {
    if (std::abs(coeff) > 1e-12) { // Игнорируем почти нулевые
        coeffs[power] += coeff;
        if (std::abs(coeffs[power]) < 1e-12) {
            coeffs.erase(power);
        }
    }
}

// Вывод на экран
void LaurentSeries::print() const {
    if (coeffs.empty()) {
        std::cout << "0" << std::endl;
        return;
    }
    
    bool first = true;
    // Итерируемся от старшей степени к младшей
    for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
        int p = it->first;
        std::complex<double> c = it->second;
        
        // Если коэффициент чисто вещественный, выводим проще
        bool is_real = (std::abs(c.imag()) < 1e-12);
        
        if (!first) {
            if (is_real && c.real() > 0) {
                std::cout << " + ";
            } else if (is_real && c.real() < 0) {
                std::cout << " - ";
            } else {
                std::cout << " + ";
            }
        } else if (is_real && c.real() < 0) {
            std::cout << "-";
        }
        
        // Вывод коэффициента
        if (is_real) {
            double abs_c = std::abs(c.real());
            if (abs_c != 1.0 || p == 0) {
                std::cout << abs_c;
            }
        } else {
            // Комплексный коэффициент
            if (p == 0 || std::abs(c) != 1.0) {
                std::cout << "(" << c.real();
                if (c.imag() >= 0) std::cout << "+";
                std::cout << c.imag() << "i)";
            }
        }
        
        // Вывод степени z
        if (p != 0) {
            if (!(is_real && std::abs(c.real()) == 1.0)) {
                std::cout << "*";
            }
            std::cout << "z";
            if (p != 1) std::cout << "^" << p;
        }
        
        first = false;
    }
    std::cout << std::endl;
}

// Дифференцирование: d/dz (a_n * z^n) = n * a_n * z^(n-1)
LaurentSeries LaurentSeries::differentiate() const {
    LaurentSeries deriv;
    for (const auto& term : coeffs) {
        int n = term.first;
        std::complex<double> a = term.second;
        
        if (n != 0) { // Константа убивается
            deriv.addTerm(n - 1, a * static_cast<double>(n));
        }
    }
    return deriv;
}

// Вычислить значение ряда в точке z
std::complex<double> LaurentSeries::evaluate(std::complex<double> z) const {
    std::complex<double> sum(0.0, 0.0);
    for (const auto& term : coeffs) {
        int power = term.first;
        std::complex<double> coeff = term.second;
        // coeff * z^power
        sum += coeff * std::pow(z, power);
    }
    return sum;
}

// Поиск корней (комплексных)
std::vector<std::complex<double>> LaurentSeries::findRoots() const {
    std::vector<std::complex<double>> roots;
    if (coeffs.empty()) return roots;

    // 1. Определяем реальные границы ряда
    int min_deg = coeffs.begin()->first;
    int max_deg = coeffs.rbegin()->first;
    
    if (min_deg > 0) {
        for (int i = 0; i < min_deg; ++i) {
            roots.push_back(std::complex<double>(0.0, 0.0));
        }
    }

    if (coeffs.size() == 1) {
        return roots; // Уже добавили все корни в нуле
    }

    if (min_deg == max_deg) {
        return roots;
    }

    if (min_deg == max_deg && min_deg == 0) return {};

    // 2. Превращаем в Полином P(z).
    // Домножаем ряд на z^(-min_deg), чтобы младшая степень стала 0.
    int N = max_deg - min_deg;

    // Вектор коэффициентов полинома P(z) = p_0 + p_1*z + ... + p_N*z^N
    Eigen::VectorXcd poly_coeffs = Eigen::VectorXcd::Zero(N + 1);

    for (auto const& [power, val] : coeffs) {
        int idx = power - min_deg;
        if (idx >= 0 && idx <= N) {
            poly_coeffs(idx) = val;
        }
    }

    // Нормализуем, чтобы коэффициент при старшей степени (pN) был равен 1
    std::complex<double> leading_coeff = poly_coeffs(N);

    if (std::abs(leading_coeff) < 1e-12) {
        return roots; // Вырожденный случай
    }

    poly_coeffs /= leading_coeff;

    // 3. Строим Сопровождающую матрицу (Companion Matrix) размера N x N
    Eigen::MatrixXcd companion(N, N);
    companion.setZero();

    // Заполняем поддиагональ единицами
    if (N > 1) {
        for (int i = 0; i < N - 1; ++i) {
            companion(i + 1, i) = 1.0;
        }
    }
     
    // Последний столбец содержит коэффициенты с минусом
    // P(z) = z^N + a_{N-1}z^{N-1} + ... + a_0
    // Companion: Последний столбец: -a_0, -a_1, ..., -a_{N-1}
    for (int i = 0; i < N; ++i) {
        companion(i, N - 1) = -poly_coeffs(i);
    }

    // 4. Находим собственные числа
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(companion);
    
    // Результат в виде вектора комплексных чисел
    Eigen::VectorXcd eigenvalues = solver.eigenvalues();

    // Добавляем ненулевые корни
    for (int i = 0; i < eigenvalues.size(); ++i) {
        roots.push_back(eigenvalues(i));
    }

    return roots;
}




































































































