#!/bin/bash

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Compilation script for Mandelbrot/Julia fractal project with CUDA support"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME    Set output executable name (default: 'mandelbrot')"
    echo "  -a, --all          Force recompile all source files, including LaurentSeries.cpp"
    echo "  -h, --help         Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0                    # Default compilation"
    echo "  $0 --name fractal     # Compile with custom output name"
    echo "  $0 --all              # Force recompile all files"
    echo "  $0 --all --name app   # Force recompile with custom name"
    echo ""
    echo "Notes:"
    echo "  - LaurentSeries.cpp is compiled only if LaurentSeries.o doesn't exist"
    echo "    or when using --all flag"
    echo "  - Uses CUDA compiler (nvcc) for GPU kernels"
    echo "  - Requires Eigen3 library and CUDA toolkit"
    echo ""
    echo "Environment:"
    echo "  CUDA_PATH: /usr/local/cuda"
    echo "  Eigen3: /usr/include/eigen3"
    exit 0
}

OUTPUT_NAME="mandelbrot"
FORCE_COMPILE_ALL=false
COMPILE_LAURENT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            if [ -z "$2" ]; then
                echo "Error: --name requires an argument"
                exit 1
            fi
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -a|--all)
            FORCE_COMPILE_ALL=true
            shift
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done


echo "Compile to: $OUTPUT_NAME"

if [ -f "LaurentSeries.o" ] && [ "$FORCE_COMPILE_ALL" = false ]; then
    echo "LaurentSeries.o found, skipping..."
    COMPILE_LAURENT=false
else
    COMPILE_LAURENT=true
fi

echo "Compiling..."

g++ -c helpers.cpp -O3 -I /usr/include/eigen3 -fPIC &
nvcc -c main.cu -I /usr/include/eigen3 -D EIGEN_NO_CUDA &
nvcc -c kernels.cu -I /usr/include/eigen3 -D EIGEN_NO_CUDA &

if [ "$COMPILE_LAURENT" = true ]; then
    g++ -c LaurentSeries.cpp -O3 -I /usr/include/eigen3 -fPIC &
fi

wait

echo "Linking..."
g++ main.o kernels.o helpers.o LaurentSeries.o -o "$OUTPUT_NAME" -lz -lcudart -L/usr/local/cuda/lib64 -no-pie

if [ $? -eq 0 ]; then
    echo "Compilation successful: $OUTPUT_NAME"
else
    echo "Compilation failed"
    exit 1
fi
