#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <stdint.h>
#include "fp8_utils.h"

// This code benchmarks a SWAR (SIMD Within A Register) implementation of FP8 
// multiplication against a scalar implementation that processes one FP8 multiplication 
// at a time.

#ifndef NUM_OPERATIONS
#define NUM_OPERATIONS 50'000'000 // 50 million packed registers (200Many FP8s)
#endif


// Helper to pin the process to a specific core to reduce noise in benchmarking
void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if(sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
    {
        perror("sched_setaffinity");
    }
}

// Helper to validate randomly generated FP8 values based on E5M2 requirements
bool is_valid_fp8(uint8_t val) {
    uint8_t mag = val & 0x7F;
    if (mag > 0x7C) return false; // Values above X1111100 in magnitude are invalid
    if (mag > 0x00 && mag < 0x04) return false; // Zero exponent and non-null mantissa are invalid
    return true;
}

// Evaluates all 4 lanes of a 32-bit register to ensure every packed FP8 value satisfies 
// coherence rules
bool is_coherent_32(uint32_t val) {
    for (int i = 0; i < 4; ++i) if (!is_valid_fp8((val >> (i * 8)) & 0xFF)) return false;
    return true;
}

// Evaluates all 8 lanes of a 64-bit register to ensure every packed FP8 value satisfies 
// coherence rules
bool is_coherent_64(uint64_t val) {
    for (int i = 0; i < 8; ++i) if (!is_valid_fp8((val >> (i * 8)) & 0xFF)) return false;
    return true;
}


int main() {


    pin_to_core(0); // Pin to core 0 for more consistent benchmarking

    // Initialize Mersenne Twister RNG with fixed seed (43) for reproducible results
    std::mt19937 rng(43); 

    
    #ifdef TEST_32

    // Defining the data for 32-bit tests and sinks to prevent optimization
    // 'volatile' forces the compiler to write this to RAM, preventing it 
    // from deleting our loops as "dead code".
    std::vector<uint32_t> op1_data_32(NUM_OPERATIONS);
    std::vector<uint32_t> op2_data_32(NUM_OPERATIONS);
    std::vector<uint8_t> op1_data_32_8(NUM_OPERATIONS * 4);
    std::vector<uint8_t> op2_data_32_8(NUM_OPERATIONS * 4);
    volatile uint32_t sink_32 = 0;
    volatile uint32_t sink_32_8 = 0;

    // Defining variables to hold timing results for 32-bit tests
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_32;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_32;
    std::chrono::duration<double> time_scalar_32;
    std::chrono::duration<double> time_swar_32;

    // Fill with random garbage data (bypassing zero-optimizations)
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        do {
            op1_data_32[i] = rng();
        } while (!is_coherent_32(op1_data_32[i]));
        do {
            op2_data_32[i] = rng();
        } while (!is_coherent_32(op2_data_32[i]));
        // Unpack the valid 32-bit SWAR data into individual 8-bit arrays for the scalar 
        // for-loop benchmark
        for (size_t j = 0; j < 4; ++j) {
            op1_data_32_8[i*4 + j] = (op1_data_32[i] >> (j*8)) & 0xFF;
            op2_data_32_8[i*4 + j] = (op2_data_32[i] >> (j*8)) & 0xFF;
        }
    }

    // Testing Scalar Implementation for 32-bit registers (processing 4 FP8s at a time)

    // If compiled with the TEST_8_FOR_LOOP flag, we will do the scalar multiplication 
    // with a simple for loop
    #ifdef TEST_8_FOR_LOOP
    start_time_32 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 4 * NUM_OPERATIONS; ++i) {
        sink_32_8 = scalar_fp8_mul(op1_data_32_8[i], op2_data_32_8[i]);
    }
    end_time_32 = std::chrono::high_resolution_clock::now();
    time_scalar_32 = end_time_32 - start_time_32;
    
    // If not, we will do simple scalar multiplications by extracting fp8 from each lane
    // with bit shifts and masking, doing the multiplication, and then repacking the 
    // result. 
    #else
    start_time_32 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        sink_32 = process_4_scalars(op1_data_32[i], op2_data_32[i]);
    }
    end_time_32 = std::chrono::high_resolution_clock::now();
    time_scalar_32 = end_time_32 - start_time_32;
    #endif


    // Testing SWAR Implementation for 32-bit registers (processing 4 FP8s at a time)
    start_time_32 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        sink_32 = swar_fp8_mul(op1_data_32[i], op2_data_32[i]);
    }
    end_time_32 = std::chrono::high_resolution_clock::now();
    time_swar_32 = end_time_32 - start_time_32;

    // Printing results for 32-bit tests
    std::cout << "Processed " << NUM_OPERATIONS * 4 << " FP8 Multiplications.\n";
    std::cout << "Scalar Time: " << time_scalar_32.count() << " seconds\n";
    std::cout << "SWAR Time:   " << time_swar_32.count() << " seconds\n";
    std::cout << "\nSWAR is " << (time_scalar_32.count() / time_swar_32.count()) << "x faster.\n";

    // Check that lanes are independent in the SWAR implementation by comparing with 
    // the scalar implementation on each lane separately. 
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        uint32_t res1 = swar_fp8_mul(op1_data_32[i], op2_data_32[i]);
        uint32_t res2 = process_4_scalars(op1_data_32[i], op2_data_32[i]);
        if (res1 != res2) {
            std::cerr << "Final Check Mismatch at index " << i 
                      << " | Scalar: 0x" << std::hex << res2 
                      << " | SWAR: 0x" << std::hex << res1 << std::dec << "\n"
                      << " Operands: op1: 0x" << std::hex << op1_data_32[i] 
                      << " op2: 0x" << std::hex << op2_data_32[i] << std::dec << "\n";
            return -1;
        }
    }

    #endif // TEST_32
    


    #ifdef TEST_64

    // Defining the data for 64-bit tests and sinks to prevent optimization
    // 'volatile' forces the compiler to write this to RAM, preventing it 
    // from deleting our loops as "dead code".
    std::vector<uint64_t> op1_data_64(NUM_OPERATIONS);
    std::vector<uint64_t> op2_data_64(NUM_OPERATIONS);
    std::vector<uint8_t> op1_data_64_8(NUM_OPERATIONS * 8);
    std::vector<uint8_t> op2_data_64_8(NUM_OPERATIONS * 8);
    volatile uint64_t sink_64 = 0;
    volatile uint64_t sink_64_8 = 0;

    // Defining variables to hold timing results for 64-bit tests
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_64;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_64;
    std::chrono::duration<double> time_scalar_64;
    std::chrono::duration<double> time_swar_64;

    // Fill with random garbage data (bypassing zero-optimizations)
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        do {
            // rng() produces 32-bit integers, so we shift and bitwise OR two calls to 
            // fill 64 bits
            op1_data_64[i] = ((uint64_t)rng() << 32) | rng();
        } while (!is_coherent_64(op1_data_64[i]));
        do {
            op2_data_64[i] = ((uint64_t)rng() << 32) | rng();
        } while (!is_coherent_64(op2_data_64[i]));
        // Unpack the valid 64-bit SWAR data into individual 8-bit arrays for the scalar 
        // for-loop benchmark
        for (size_t j = 0; j < 8; ++j) {
            op1_data_64_8[i*8 + j] = (op1_data_64[i] >> (j*8)) & 0xFF;
            op2_data_64_8[i*8 + j] = (op2_data_64[i] >> (j*8)) & 0xFF;
        }
    }

    // Testing Scalar Implementation for 64-bit registers (processing 8 FP8s at a time)
    
    // If compiled with the TEST_8_FOR_LOOP flag, we will do the scalar multiplication
    // with a simple for loop
    #ifdef TEST_8_FOR_LOOP
    start_time_64 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 8 * NUM_OPERATIONS; ++i) {
        sink_64_8 = scalar_fp8_mul(op1_data_64_8[i], op2_data_64_8[i]);
    }
    end_time_64 = std::chrono::high_resolution_clock::now();
    time_scalar_64 = end_time_64 - start_time_64;

    // If not, we will do simple scalar multiplications by extracting fp8 from each lane
    // with bit shifts and masking, doing the multiplication, and then repacking the 
    // result.
    #else
    start_time_64 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        sink_64 = process_8_scalars(op1_data_64[i], op2_data_64[i]);
    }
    end_time_64 = std::chrono::high_resolution_clock::now();
    time_scalar_64 = end_time_64 - start_time_64;
    #endif

    // Testing SWAR Implementation for 64-bit registers (processing 8 FP8s at a time)
    start_time_64 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        sink_64 = swar_fp8_mul(op1_data_64[i], op2_data_64[i]);
    }
    end_time_64 = std::chrono::high_resolution_clock::now();
    time_swar_64 = end_time_64 - start_time_64;

    // Printing results for 64-bit tests
    std::cout << "Processed " << NUM_OPERATIONS * 8 << " FP8 Multiplications.\n";
    std::cout << "Scalar Time: " << time_scalar_64.count() << " seconds\n";
    std::cout << "SWAR Time:   " << time_swar_64.count() << " seconds\n";
    std::cout << "\nSWAR is " << (time_scalar_64.count() / time_swar_64.count()) << "x faster.\n";

    // Check that lanes are independent in the SWAR implementation by comparing with 
    // the scalar implementation on each lane separately. 
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        uint64_t res1 = swar_fp8_mul(op1_data_64[i], op2_data_64[i]);
        uint64_t res2 = process_8_scalars(op1_data_64[i], op2_data_64[i]);
        if (res1 != res2) {
            std::cerr << "Final Check Mismatch at index " << i 
                      << " | Scalar: 0x" << std::hex << res2 
                      << " | SWAR: 0x" << std::hex << res1 << std::dec << "\n"
                      << " Operands: op1: 0x" << std::hex << op1_data_64[i] 
                      << " op2: 0x" << std::hex << op2_data_64[i] << std::dec << "\n";
            return -1;
        }
    }

    #endif // TEST_64

    return 0;

}