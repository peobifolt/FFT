#ifndef TEST_FASTFOURIERTRANSFORM_H
#define TEST_FASTFOURIERTRANSFORM_H


#include <vector>
#include <complex>
#include <cassert>
#include <iostream>

template<typename DOUBLE_TYPE>
class FastFourierTransform {
    using complex_t = std::complex<DOUBLE_TYPE>;

    /**
     * Returns true if n = 2^k * 3^m * 5^p.
     * @param n a number
     * @return condition for n
     */
    static bool only235(size_t n) {
        while (n % 2 == 0) {
            n /= 2;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        while (n % 5 == 0) {
            n /= 5;
        }
        return n == 1;
    }

    /**
     * Complements the array a with zeros up to the number n, in the decomposition of
     * which only prime 2, 3 and 5 occur
     * @param a an array
     */
    static void extend(std::vector<complex_t> &a) {
        size_t n = a.size();
        while (!only235(n)) {
            n++;
        }
        a.resize(n);
    }

    /**
     * Returns exp(2 * pi * i / n).
     * @param n a parameter
     * @return the value
     */
    static complex_t get_w(size_t n) {
        DOUBLE_TYPE angle = std::numbers::pi * 2 / n;
        return complex_t(cos(angle), sin(angle));
    }

public:
    /**
     * Does the Fourier transform. Extends array {@code a} for 2^n * 3^m * 5^k.
     * Result puts in array {@code a}.
     * @param a a polynomial
     */
    static void fft(std::vector<complex_t> &a) {
        extend(a);
        size_t n = a.size();
        for (size_t p : {2, 3, 5}) {
            while (n % p == 0) {
                std::vector<complex_t> tmp(n);
                for (size_t i = 0; i < a.size(); i += n) {
                    for (size_t j = 0; j < p; j++) {
                        for (size_t k = 0; k * p < n; k++) {
                            tmp[j * (n / p) + k] = a[i + k * p + j];
                        }
                    }
                    copy(tmp.begin(), tmp.end(), a.begin() + i);
                }
                n /= p;
            }
        }
        for (size_t p : {5, 3, 2}) {
            while (a.size() % (p * n) == 0) {
                complex_t w_n = get_w(n * p);
                complex_t w_step_n = 1;  // w^n
                for (size_t i = 0; i < n; i++) {
                    w_step_n *= w_n;
                }
                std::vector<complex_t> tmp(p);
                for (size_t i = 0; i < a.size(); i += n * p) {
                    complex_t w_step_j = 1;  // w^j
                    for (size_t j = 0; j < n; j++) {
                        fill(tmp.begin(), tmp.end(), 0);
                        complex_t w_step_nk = 1;  // w^(nk)
                        for (size_t k = 0; k < p; k++) {
                            complex_t w_step_j_plus_nk = w_step_j * w_step_nk;  // w^(j + nk)
                            complex_t w_step_s_mul_j_plus_nk = 1;  // w^(s*(j + nk))
                            for (size_t s = 0; s < p; s++) {
                                tmp[k] += a[i + j + s * n] * w_step_s_mul_j_plus_nk;
                                w_step_s_mul_j_plus_nk *= w_step_j_plus_nk;
                            }
                            w_step_nk *= w_step_n;
                        }
                        for (size_t k = 0; k < p; k++) {
                            a[i + j + k * n] = tmp[k];
                        }
                        w_step_j *= w_n;
                    }
                }
                n *= p;
            }
        }
    }

    /**
     * Does the invert Fourier transform. Extends array {@code a} for 2^n * 3^m * 5^k.
     * Result puts in array {@code a}.
     * @param a a polynomial
     */
    static void ifft(std::vector<complex_t> &a) {
        fft(a);
        reverse(a.begin() + 1, a.end());
        for (complex_t &i: a) {
            i /= a.size();
        }
    }

    /**
     * Multiples two non empty complex polynomials.
     * @param a first polynomial
     * @param b second polynomial
     * @return result multiplication
     */
    static std::vector<complex_t> convolution(std::vector<complex_t> a, std::vector<complex_t> b) {
        assert(!a.empty() && !b.empty());
        size_t len = a.size() + b.size();
        a.resize(len);
        b.resize(len);
        fft(a);
        fft(b);
        for (size_t i = 0; i < a.size(); i++) {
            a[i] *= b[i];
        }
        ifft(a);
        a.resize(len - 1);
        return a;
    }
};


#endif //TEST_FASTFOURIERTRANSFORM_H
