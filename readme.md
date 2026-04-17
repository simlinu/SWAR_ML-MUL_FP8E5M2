# FP8 L-MUL SWAR Optimization

This project provides an optimized software emulation for 8-bit floating-point (FP8 E5M2) multiplication using Logarithmic Multiplication (L-MUL) combined with SIMD Within A Register (SWAR) techniques.

## Benchmarking (`benchmark.cpp`)

The file `benchmark.cpp` contains the `main` function to test the execution speedup of the SWAR kernels against a standard scalar baseline. 

It must be compiled with specific macros to determine the architecture width and baseline behavior:

* **`-DTEST_32`**: Compiles and tests the 32-bit SWAR implementation, which processes 4 FP8 multiplications simultaneously within a single 32-bit register.
* **`-DTEST_64`**: Compiles and tests the 64-bit SWAR implementation, which processes 8 FP8 multiplications simultaneously within a single 64-bit register.
* **`-DTEST_8_FOR_LOOP`**: By default, the baseline scalar benchmark simulates unpacking data from a packed register, multiplying sequentially, and repacking the results. Defining this flag modifies the baseline to instead perform a simple `for` loop of scalar FP8 multiplications directly from pre-unpacked 8-bit arrays.

### Example Compilation
```bash
g++ -O3 -DTEST_64 -DTEST_8_FOR_LOOP benchmark.cpp fp8_utils.cpp -o benchmark
```

## Correctness Testing (`ml_mull_correctness.cpp`)

The file `ml_mull_correctness.cpp` is isolated from the performance benchmarks and is only used to test the mathematical correctness of the FP8 L-MUL approximation. 

It iterates through possible E5M2 bit patterns to verify that our logarithmic addition correctly matches native hardware multiplication under different precision constraints, specifically validating our emulation's truncation boundaries against standard round-to-nearest-even (RNE) rounding.
