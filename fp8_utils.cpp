#include <cstdint>
#include "fp8_utils.h"

// ---------------------------------------------------------
// SWAR Implementation (Processing 8 at a time)
// ---------------------------------------------------------
uint64_t swar_fp8_mul(uint64_t a, uint64_t b) {

    // 0. NaN Detection
    // We want to detect whether one of the two inputs is NaN to set the result to NaN 
    // in that case. In E5M2 NaN is represented by 0x80 (b10000000, the sign bit is 1 
    // and the rest is 0). So we can detect whether a lane is NaN by checking whether 
    // the sign bit is 1 and the rest of the bits are 0. 
    // We can achieve this by subtracting from the value its 8th bit shifted seven times
    // to the right so that the result lane will have 0 in the eighth bit if it was NaN
    // or if it had the 8th bit set to 0. 
    // To exclude the case in which the 8th bit was set to 0 we and the negation of the 
    // result with the original value so that we will have 0 in eigth bit  if it was not 
    // NaN and 1 if it was NaN. Then we mask the other bits to 0.
    uint64_t a_bit = a - ((a & 0x8080808080808080ULL) >> 7);
    uint64_t a_nan = (a & ~a_bit) & 0x8080808080808080ULL; 
    uint64_t b_bit = b - ((b & 0x8080808080808080ULL) >> 7);
    uint64_t b_nan = (b & ~b_bit) & 0x8080808080808080ULL;
    uint64_t nan_bit = a_nan | b_nan; 
    
    // 0.5 NaN Masks
    // We have to use two masks, the first sets everything to 0 and the other sets the 
    // sign bit to 1. 
    // 0xFF if nan and 0x00 otherwise, we will AND it with the final result at the end 
    // to set to NaN the lanes where there was a NaN in input
    uint64_t nan_mask1 = ~(nan_bit - (nan_bit >> 7)); 

    // 0x80 if nan and 0x00 otherwise, we will OR it with the final result at the end to 
    // set to NaN the lanes where there was a NaN in input
    uint64_t nan_mask2 = nan_bit;


    // 1. Extract Signs
    // XOR the sign bits together to get the final signs and mask the non sign bits. So 
    // we have for each fp8 lane 0x00 (b00000000) if both inputs had the same sign and 
    // the result sign is positive, and 0x80 (b10000000) if they had different signs
    // and the result sign is negative
    uint64_t signs = (a ^ b) & 0x8080808080808080ULL;

    // 2. Add Magnitudes (Exponent + Mantissa)
    // In each lane we treat the concatenation of exponent and mantissa as a 7-bit 
    // unsigned number and we add them together. Since the last bit is not used as the
    // sign bit we can use it to hold the carry of the sum, which allows us to detect 
    // overflows and underflows without branching. Max value per lane is 127 + 127 = 254 
    // Carry-over among lanes is mathematically impossible (of course we have to set 
    // to zero the sign bits before to do the addition to avoid interferences).
    uint64_t sum = (a & 0x7F7F7F7F7F7F7F7FULL) + (b & 0x7F7F7F7F7F7F7F7FULL);

    // 3. Safe Subtraction of 0x3C (The Bias)
    // We want to compute (sum - 60) for each lane and we need to do it in a way that 
    // detects and handles underflows and overflows without branching and avoids 
    // interferences among lanes. The idea is to set the eighth bit to 1 and subtract 
    // so that if there is no underflow the result is correct and if there is an 
    // underflow the eighth bit is 0  
    uint64_t half_add = (0x8080808080808080ULL | sum) - 0x3C3C3C3C3C3C3C3CULL;

    // 4. Overflow detection
    // If the result is greater or equal to 0x7F (b01111111) then it has overflown. We 
    // add to half_add 0x01 (b00000001) so that it carries over to the eigth bit
    // If 8th bit is 1 both in sum and half_add', it means there is overflow and we store 
    // it in a mask with 0x80 (b10000000) in the lanes where there was an overflow and 
    // 0x00 (b00000000) where there was not.
    // We cannot check only the 8th bit of the half_add' lanes because if the initial 
    // sum did not carry over the 7 bits it would be a false positive overflow. For 
    // example, if sum is equal 0x40 (b01000000) and we subtract 0x3C (b00111100) we get 
    // 0x04 (b00000100) without overflow. However, half_add would have the 8th bit set 
    // to 1 because we set it to 1 before the subtraction.
    uint64_t overflow_bit = sum & (half_add + 0x0101010101010101ULL) & 0x8080808080808080ULL;

    // 5. Overflow mask
    // When we have an overflow the result should be saturated to 0x7F (b01111111, the 
    // max valid E5M2 value). 
    // We can achieve this by creating a mask with 0x7F (b01111111) in the lanes where 
    // there was an overflow and 0x00 (b00000000) otherwise
    // The  mask is created by smearing the 8th bit: we subtract the value with its 
    // right-shifted-by-7 version so that if there was an overflow we get 0x7F 
    // (b01111111) and if there was not we get 0x00 (b00000000)
    uint64_t overflow_mask = overflow_bit - (overflow_bit >> 7);

    // 6. Underflow detection
    // We have underflow if result is smaller than 0x01 (b00000001). So we subtract from
    // half_add 0x01 (b00000001) so that we are sure that it will set to 0 the eight bit
    // If 8th bit is 0 both in sum and half_add', it means there is underflow and we 
    // store it in a mask with 0x80 (b10000000) in the lanes where there was an 
    // underflow and 0x00 (b00000000) where there was not.
    // We cannot check only the 8th bit of the half_add' lanes because if the initial 
    // sum carried over the 7 bits it would be a false positive underflow. For example
    // if sum is equal 0x80 (b10000000) and we subtract 0x3C (b00111100) we get 0x40
    // (b01000100) without underflow. However, half_add would have the 8th bit set 
    // to 0
    uint64_t underflow_bit = ~sum & ~(half_add - 0x0101010101010101ULL) & 0x8080808080808080ULL;

    // 6.1 Zero Operand Detection 
    // When one operand is zero the multiplication has to return zero
    // We check whether one operand is zero by inverting all the bits, masking the first
    // seven and then adding one, in this way only if all the seven bits are set to zero
    // the eighth bit will be set to 1.
    // Then we simply OR it with the underflow bit as the expect behavior onwards is the
    // same
    uint64_t a_zero = ((~a & 0x7F7F7F7F7F7F7F7FULL) + 0x0101010101010101ULL) & 0x8080808080808080ULL;
    uint64_t b_zero = ((~b & 0x7F7F7F7F7F7F7F7FULL) + 0x0101010101010101ULL) & 0x8080808080808080ULL;
    underflow_bit = underflow_bit | a_zero | b_zero;


    // 7. Underflow mask
    // When we have an underflow the result should be flushed to zero. We can achieve
    // this by creating a mask with 0x00 (b00000000) in the lanes where there was an 
    // underflow and 0xFF (b11111111) where there was not and ANDing the result with it 
    // (setting all bits to 0 if there was an underflow and leaving it unchanged 
    // otherwise).
    // We create the mask by smearing the 8th bit and readding it: we subtract the 
    // value with its right-shifted-by-7 version so that if there was an underflow we 
    // get 0x00 (b00000000) and if there was not we get 0xFF (b11111111)
    uint64_t underflow_mask = ~(underflow_bit - (underflow_bit >> 7) + underflow_bit);

    // 8. Final result 
    // Now we have all the components to compute the final result:
    // We apply the overflow mask to the half_add so that if there was an overflow we 
    // set the result to 0x7C (b01111111, the max valid E5M2 value),
    // Then we combine the signs with the 7-bit result (we have to mask the signs to 
    // avoid interferences) by ORing them together 
    // Then we apply the underflow mask so that if there was an underflow we set the
    // result to 0x00 (b00000000) 
    // And finally we apply the NaN Mask setting the result to 0x80 (b10000000) if the 
    // result is NaN (overflow with negative sign) and leaving it
    uint64_t res_7bit = (half_add | overflow_mask);
    uint64_t res_bef_uf_nan = signs | (res_7bit & 0x7F7F7F7F7F7F7F7FULL);
    uint64_t res_bef_nan = res_bef_uf_nan & underflow_mask;
    uint64_t res = (res_bef_nan & nan_mask1) | nan_mask2;

    return res;

}

// ---------------------------------------------------------
// SWAR Implementation (Processing 4 at a time)
// ---------------------------------------------------------
uint32_t swar_fp8_mul(uint32_t a, uint32_t b) {
    
    // For the documentation of the algorithm please refer to the 8 at a time version, 
    // the only difference is that here we have 4 lanes instead of 8 
    
    uint32_t a_bit = a - ((a & 0x80808080) >> 7);
    uint32_t a_nan = (a & ~a_bit) & 0x80808080; 
    uint32_t b_bit = b - ((b & 0x80808080) >> 7);
    uint32_t b_nan = (b & ~b_bit) & 0x80808080;
    uint32_t nan_bit = a_nan | b_nan; 
    
    uint32_t nan_mask1 = ~(nan_bit - (nan_bit >> 7)); 
    uint32_t nan_mask2 = nan_bit;

    uint32_t signs = (a ^ b) & 0x80808080;
    uint32_t sum = (a & 0x7F7F7F7F) + (b & 0x7F7F7F7F);
    uint32_t half_add = (0x80808080 | sum) - 0x3C3C3C3C; 
    uint32_t overflow_bit = sum & (half_add + 0x01010101) & 0x80808080; 
    uint32_t overflow_mask = overflow_bit - (overflow_bit >> 7);
    uint32_t underflow_bit = ~sum & ~(half_add - 0x01010101) & 0x80808080;
    uint32_t a_zero = ((~a & 0x7F7F7F7F) + 0x01010101) & 0x80808080;
    uint32_t b_zero = ((~b & 0x7F7F7F7F) + 0x01010101) & 0x80808080;
    underflow_bit = underflow_bit | a_zero | b_zero;
    uint32_t underflow_mask = ~(underflow_bit - (underflow_bit >> 7) + underflow_bit); 
    
    uint32_t res_7bit = (half_add | overflow_mask);
    uint32_t res_bef_uf_nan = signs | (res_7bit & 0x7F7F7F7F);
    uint32_t res_bef_nan = res_bef_uf_nan & underflow_mask;
    uint32_t res = (res_bef_nan & nan_mask1) | nan_mask2;

    return res;
}

// ---------------------------------------------------------
// The Scalar Implementation (Processing 1 at a time)
// ---------------------------------------------------------
uint8_t scalar_fp8_mul(uint8_t a, uint8_t b) {

    // The algorithm is the same as the one used in the SWAR versions, but here we only 
    // have 1 lane so refer to that for the documentation of the steps. 

    uint8_t a_bit = a - ((a & 0x80) >> 7);
    uint8_t a_nan = (a & ~a_bit) & 0x80; 
    uint8_t b_bit = b - ((b & 0x80) >> 7);
    uint8_t b_nan = (b & ~b_bit) & 0x80;
    uint8_t nan_bit = a_nan | b_nan; 
    
    uint8_t nan_mask1 = ~(nan_bit - (nan_bit >> 7)); 
    uint8_t nan_mask2 = nan_bit;

    uint8_t signs = (a ^ b) & 0x80;
    uint8_t sum = (a & 0x7F) + (b & 0x7F);
    uint8_t half_add = (0x80 | sum) - 0x3C; // 0x3C is the bias for E5M2
    uint8_t overflow_bit = sum & (half_add + 0x01) & 0x80;
    uint8_t overflow_mask = overflow_bit - (overflow_bit >> 7);
    uint8_t underflow_bit = ~sum & ~(half_add - 0x01) & 0x80;
    uint8_t a_zero = ((~a & 0x7F) + 0x01) & 0x80;
    uint8_t b_zero = ((~b & 0x7F) + 0x01) & 0x80;
    underflow_bit = underflow_bit | a_zero | b_zero;
    uint8_t underflow_mask = ~(underflow_bit - (underflow_bit >> 7) + underflow_bit);

    uint8_t res_7bit = (half_add | overflow_mask);
    uint8_t res_bef_uf_nan = signs | (res_7bit & 0x7F);
    uint8_t res_bef_nan = res_bef_uf_nan & underflow_mask;
    uint8_t res = (res_bef_nan & nan_mask1) | nan_mask2;

    return res;

}

// Helper to simulate unpacking, doing 4 scalar ops, and repacking
uint32_t process_4_scalars(uint32_t a, uint32_t b) {

    uint8_t res0 = scalar_fp8_mul((a >> 0) & 0xFF, (b >> 0) & 0xFF);
    uint8_t res1 = scalar_fp8_mul((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    uint8_t res2 = scalar_fp8_mul((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    uint8_t res3 = scalar_fp8_mul((a >> 24) & 0xFF, (b >> 24) & 0xFF);
    
    return (res3 << 24) | (res2 << 16) | (res1 << 8) | res0;
}

// Helper to simulate unpacking, doing 8 scalar ops, and repacking
uint64_t process_8_scalars(uint64_t a, uint64_t b) {
    uint8_t res0 = scalar_fp8_mul((a >> 0) & 0xFF, (b >> 0) & 0xFF);
    uint8_t res1 = scalar_fp8_mul((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    uint8_t res2 = scalar_fp8_mul((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    uint8_t res3 = scalar_fp8_mul((a >> 24) & 0xFF, (b >> 24) & 0xFF);
    uint8_t res4 = scalar_fp8_mul((a >> 32) & 0xFF, (b >> 32) & 0xFF);
    uint8_t res5 = scalar_fp8_mul((a >> 40) & 0xFF, (b >> 40) & 0xFF);
    uint8_t res6 = scalar_fp8_mul((a >> 48) & 0xFF, (b >> 48) & 0xFF);
    uint8_t res7 = scalar_fp8_mul((a >> 56) & 0xFF, (b >> 56) & 0xFF);
    
    return ((uint64_t)res7 << 56) | ((uint64_t)res6 << 48) | ((uint64_t)res5 << 40) | 
           ((uint64_t)res4 << 32) | ((uint64_t)res3 << 24) | ((uint64_t)res2 << 16) | 
           ((uint64_t)res1 << 8) | (uint64_t)res0;
}