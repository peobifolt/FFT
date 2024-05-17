# Пробное задание на стажировку в Yadro. FFT
### Условие:
1. Написать на С++ класс быстрого прямого и обратного преобразования
Фурье комплексных значений с возможной длиной преобразования
кратной 2, 3, 5.
2. Запустить для случайных комплексных входных данных сначала
прямое, а потом обратное преобразование Фурье.
3. Сравнить ошибку между входными и выходными данными.

Решение находится в файле [FastFourierTransform.h](https://github.com/peobifolt/FFT/blob/main/FastFourierTransform.h)
Запуск на случайном большом тесте в файле [tests.cpp](https://github.com/peobifolt/FFT/blob/main/tests.cpp) в тесте test_big_numbers.

Версия компилятора:
g++ 11.4.0
