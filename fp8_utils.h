#ifndef FP8_UTILS_H
#define FP8_UTILS_H

// SWAR Implementation (Processing 8 at a time)
uint64_t swar_fp8_mul(uint64_t a, uint64_t b);

// SWAR Implementation (Processing 4 at a time)
uint32_t swar_fp8_mul(uint32_t a, uint32_t b);

// The Scalar Implementation (Processing 1 at a time)
uint8_t scalar_fp8_mul(uint8_t a, uint8_t b);

// Helper to simulate unpacking, doing 4 scalar ops, and repacking
uint32_t process_4_scalars(uint32_t a, uint32_t b);

// Helper to simulate unpacking, doing 8 scalar ops, and repacking
uint64_t process_8_scalars(uint64_t a, uint64_t b); 

#endif

