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


// N.B. Edge cases like Inf * number or 0 * number or Inf * 0 are not handled in this 
// implementation as they would require additional bitwise operations.  
// In modern Deep Learning, values are heavily regularized. Neural networks generally 
// crash long before individual activations hit true mathematical Infinity therefore 
// adding the extra checks would not bring any improvements but only diminish throughput
//
// In the paper we could add something like this:
// While the proposed SWAR kernel successfully clamps true mathematical overflows to 
// E5M2 Infinity, it deliberately omits pre-existing Infinity propagation (e.g., $Inf 
// \times X = Inf$) to maximize computational throughput. Because modern DNN workloads 
// rely on aggressive regularization and clipping, activations rarely reach true 
// Infinity during stable inference. Bypassing the bitwise evaluation required to 
// isolate and propagate pre-existing 0x7C operands saves critical instruction cycles, 
// representing a pragmatic trade-off between strict IEEE-754 edge-case compliance and 
// raw inference latency.