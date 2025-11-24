#TODO напистаь что такое фрактал

# Название проекта

Этот проект разработан для выскэффективной генерации фракталов с использованьем C/C++ и CUDA.

## Установка

Файл 'LaurentSeries.o' является прекомпеллированной частью проекта и может быть использован как есть. 
Его можно скомпелировать 1 раз командой 

```bash
g++ -c LaurentSeries.cpp -O3 -I /usr/include/eigen3
```
И больше не трогать так как он не имеет зависимости от 'config.h'.

Файл 'config.h' содержит все параметры, которые управляют генерацией фрактала. Указанная там в данный момент конфигурация протестирована на сборке:

system: GNU/Linux Debian 12
CPU: Ryzen 5 (6 cores)
RAM: 128GB
GPU: RTX4060Ti16GB

## Использование

Для того чтобы изменения конфига вступили в силу вам потребуется перекомпилировать часть файлов.

Рекомендуемые комманды:

```bash
g++ -c helpers.cpp -O3 -I /usr/include/eigen3 -fPIC &                               
nvcc -c main.cu -I /usr/include/eigen3 -D EIGEN_NO_CUDA &
nvcc -c kernels.cu -I /usr/include/eigen3 -D EIGEN_NO_CUDA &
wait
nvcc main.o kernels.o helpers.o LaurentSeries.o -o mandelbrot
```

После этого вы можете запустить генерацию командой:

```bash
./mandelbrot```

После неё в файле output.csv вы найдете параметры f(z) в разложении ряда Лорана.
В папке output/ вы найдете сгенерированные фракталы.
