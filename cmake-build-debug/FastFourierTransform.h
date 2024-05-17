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
     * Returns true if n = 2^k, 3 * 2^n or 5 * 2^n.
     * @param n a number
     * @return condition for n
     */
    static bool only235(size_t n) {
        while (n % 2 == 0) {
            n /= 2;
        }
        return n <= 5;
    }

    /**
     * Complements the array a with zeros up to the number n, in the decomposition of
     * which only prime 2, 3 and 5 occur
     * @param a an array
     */
    static void extend(std::vector<complex_t> &a) {
        size_t n = a.size();
        while (n == 0 || !only235(n)) {
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
     * Does the Fourier transform. Extends array {@code a} for 2^n, 3 * 2^n or 5 * 2^n.
     * Result puts in array {@code a}.
     * @param a a polynomial
     */
    static void fft(std::vector<complex_t> &a) {
        extend(a);
        size_t n = a.size();
        while (n % 2 == 0) {
            std::vector<complex_t> tmp(n);
            for (size_t i = 0; i < a.size(); i += n) {
                for (size_t j = 0; j < n / 2; j++) {
                    tmp[j] = a[i + j * 2];
                    tmp[j + n / 2] = a[i + j * 2 + 1];
                }
                std::copy(tmp.begin(), tmp.end(), a.begin() + i);
            }
            n /= 2;
        }
        if (n > 1) {
            assert(n == 3 || n == 5);
            complex_t w_n = get_w(n);
            std::vector<complex_t> values(n);
            for (size_t i = 0; i < a.size(); i += n) {
                fill(values.begin(), values.end(), 0);
                complex_t w_cur = 1;
                for (size_t j = 0; j < n; j++) {
                    complex_t st = 1;
                    for (size_t k = 0; k < n; k++) {
                        values[j] += st * a[i + k];
                        st *= w_cur;
                    }
                    w_cur *= w_n;
                }
                std::copy(values.begin(), values.end(), a.begin() + i);
            }
        }
        for (; n < a.size(); n *= 2) {
            complex_t w_n = get_w(n * 2);
            for (size_t i = 0; i < a.size(); i += n * 2) {
                complex_t st = 1;
                for (size_t j = 0; j < n; j++) {
                    complex_t u = a[i + j];
                    complex_t v = a[i + j + n] * st;
                    a[i + j] = u + v;
                    a[i + j + n] = u - v;
                    st *= w_n;
                }
            }
        }
    }

    /**
     * Does the invert Fourier transform. Extends array {@code a} for 2^n, 3 * 2^n or 5 * 2^n.
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
