#include "FastFourierTransform.h"
#include "googletest/googletest/include/gtest/gtest.h"

#include <cassert>
#include <random>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

template<typename T>
std::vector<T> convolution(const std::vector<T> &a, const std::vector<T> &b) {
    assert(!a.empty() && !b.empty());
    std::vector<T> res(a.size() + b.size() - 1);
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            res[i + j] += a[i] * b[j];
        }
    }
    return res;
}

template<typename T>
::testing::AssertionResult vector_equal(const std::vector<std::complex<T>> &a,
                                        const std::vector<std::complex<T>> &b) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << " lens of arrays are not equal";
    }
    const T eps = 1e-3;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(real(a[i]) - real(b[i])) > eps || std::abs(imag(a[i]) - imag(b[i])) > eps) {
            return ::testing::AssertionFailure() << a[i] << " is not equal to " << b[i];
        }
    }
    return ::testing::AssertionSuccess();
}

template<typename T>
std::vector<std::complex<T>> random_vector(size_t n, size_t M = 1000) {
    std::vector<std::complex<T>> a(n);
    for (size_t i = 0; i < n; i++) {
        a[i] = std::complex<double>(rand() % M, rand() % M);
    }
    return a;
}

TEST(test_convolution, test1) {
    std::vector<std::complex<double>> a = {1, 2, 5, 6, 6, 0, 7, 1, 5};
    std::vector<std::complex<double>> b = {7, -1, -2, 9, -11, 12};
    EXPECT_TRUE(vector_equal(FastFourierTransform<double>::convolution(a, b), convolution(a, b)));
}

TEST(test_convolution, test2) {
    size_t n = 1000;
    auto a = random_vector<long double>(n);
    auto b = random_vector<long double>(n);
    EXPECT_TRUE(vector_equal(FastFourierTransform<long double>::convolution(a, b), convolution(a, b)));
}

template<typename T>
std::vector<std::complex<T>> slow_fft(const std::vector<std::complex<T>> &a, size_t n) {
    assert(n >= a.size());
    std::vector<std::complex<T>> res(n);
    for (size_t i = 0; i < n; i++) {
        std::complex<T> st = 1;
        T angle = 2 * std::numbers::pi / n * i;
        std::complex<T> w_n(cos(angle), sin(angle));
        for (size_t j = 0; j < a.size(); j++) {
            res[i] += st * a[j];
            st *= w_n;
        }
    }
    return res;
}

TEST(test_fft_and_ifft, test1) {
    std::vector<std::complex<double>> a = {1, 2, 5, 6, 6, 0, 7, 1, 5};
    std::vector<std::complex<double>> fft_a = a;
    FastFourierTransform<double>::fft(fft_a);
    size_t n = fft_a.size();
    ASSERT_GE(n, a.size());
    EXPECT_TRUE(vector_equal(fft_a, slow_fft(a, n)));

    FastFourierTransform<double>::ifft(fft_a);
    fft_a.resize(a.size());
    EXPECT_TRUE(vector_equal(a, fft_a));
}

TEST(test_fft_and_ifft, test2) {
    size_t n = 1000;
    auto a = random_vector<long double>(n);

    std::vector<std::complex<long double>> fft_a = a;
    FastFourierTransform<long double>::fft(fft_a);
    n = fft_a.size();
    ASSERT_GE(n, a.size());
    EXPECT_TRUE(vector_equal(fft_a, slow_fft(a, n)));

    FastFourierTransform<long double>::ifft(fft_a);
    fft_a.resize(a.size());
    EXPECT_TRUE(vector_equal(a, fft_a));
}

template<typename T>
T get_absolut_error(const std::vector<std::complex<T>> &a) {
    auto b = a;
    FastFourierTransform<T>::fft(b);
    FastFourierTransform<T>::ifft(b);
    b.resize(a.size());
    T error = 0;
    for (size_t i = 0; i < a.size(); i++) {
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    return error;
}

TEST(test_absolut_error, test1_double) {
    size_t n = 1000;
    auto a = random_vector<double>(n);
    double error = get_absolut_error(a);
    std::cerr << "n = " << n << ", error = " << error << std::endl;
    ASSERT_LE(error, 1e-9);
}

TEST(test_absolut_error, test2_double) {
    size_t n = 1000000;
    auto a = random_vector<double>(n);
    double error = get_absolut_error(a);
    std::cerr << "n = " << n << ", error = " << error << std::endl;
    ASSERT_LE(error, 1e-7);
}

TEST(test_absolut_error, test1_long_double) {
    size_t n = 1000;
    auto a = random_vector<long double>(n);
    long double error = get_absolut_error(a);
    std::cerr << "n = " << n << ", error = " << error << std::endl;
    ASSERT_LE(error, 1e-9);
}

TEST(test_absolut_error, test2_long_double) {
    size_t n = 1000000;
    auto a = random_vector<long double>(n);
    long double error = get_absolut_error(a);
    std::cerr << "n = " << n << ", error = " << error << std::endl;
    ASSERT_LE(error, 1e-7);
}

TEST(test_absolut_error, test_big_numbers) {
    size_t n = 1000000;
    auto a = random_vector<long double>(n, 1e9);
    long double error = get_absolut_error(a);
    std::cerr << "n = " << n << ", error = " << error << std::endl;
    ASSERT_LE(error, 1);
}