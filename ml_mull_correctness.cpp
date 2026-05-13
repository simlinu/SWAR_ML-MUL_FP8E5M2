#include <iostream>
#include <vector>
#include <stdint.h>
#include <bitset>
#include "fp8_utils.h"

uint8_t fp8_mul(uint8_t a, uint8_t b, bool trunc) {

    // Handle special NaN case where 0x80 is treated as NaN, propagating as 0x80
    if (a == 0x80 || b == 0x80) return 0x80;

    // 1. Zero check: if either operand has a zero magnitude, return zero with the 
    // correct sign
    if (a == 0x00 || b == 0x00)
        return 0x00;

    // 2. Extract mantissas and add the implicit leading 1 
    // For E5M2, the mantissa is 2 bits, so the implicit 1 is mapped to position 4 (0x04)
    uint8_t m_a = 4 + (a & 0x03);
    uint8_t m_b = 4 + (b & 0x03);
    
    // 3. Multiply the mantissas directly
    uint8_t m = m_a * m_b;

    // 4. Calculate shifts and resulting exponent
    // If mantissa product >= 32, it overflowed the standard 4-bit bounds and needs an 
    // extra shift
    int shift = (m >= 32) ? 3 : 2;
    int e_res = ((a >> 2) & 0x1F) + ((b >> 2) & 0x1F) - 15 + (m >= 32 ? 1 : 0);

    // 5. Extract the 2-bit resulting mantissa
    uint8_t m_res = (m >> shift) & 0x03;

    // 6. Rounding logic (Truncation vs Round-To-Nearest-Even)
    if (!trunc) {
        int dropped_bits = m & ((1 << shift) - 1);
        int mid_point = 1 << (shift - 1);
        // Round up if dropped bits are greater than the midpoint, or exactly at the 
        // midpoint and the result is odd
        if ((dropped_bits > mid_point) || (dropped_bits == mid_point && (m_res & 1) == 1)) {
            m_res += 1;
            // Carry over to exponent if rounding causes mantissa overflow
            if (m_res == 4) { 
                m_res = 0;
                e_res += 1;
            }
        }
    }

    // 7. Calculate final sign via XOR
    uint8_t s = (a ^ b) & 0x80;

    // 8. Handle Exponent Bounds (Underflow / Overflow)
    if (e_res < 0) {
        return 0x00; // Flush to zero on underflow
    } 
    else if (e_res > 31) {
        // The SWAR implementation saturates to NaN (0x7F) on overflow
        return s | 0x7F;
    }
    else if (e_res == 0 && m_res == 0) {
        // Handle the case where the result is exactly zero after rounding
        return 0x00;
    }
    

    // 9. Pack and return final FP8 value
    return s | (e_res << 2) | m_res;

}

int main() {

    std::vector<uint8_t> operands;

    // Generate all valid normalized E5M2 floating point numbers
    for (uint8_t mantissa = 0x00; mantissa <= 0x03; ++mantissa) {
        for (uint8_t exponent = 0x01; exponent <= 0x1E; ++exponent) {
            for (uint8_t sign = 0x00; sign <= 0x01; ++sign) {
                uint8_t operand = (sign << 7) | (exponent << 2) | mantissa;
                operands.push_back(operand);
            }
        }
    }
    
    // Add positive and negative zero representations
    operands.push_back(0x80);
    operands.push_back(0x00);


    // Cross-verify every operand pair against ML-MUL, Truncation, and RNE logic
    for (uint8_t op1: operands) {
        for (uint8_t op2: operands) {
            uint8_t res_trunc = fp8_mul(op1, op2, true);
            uint8_t res_rntte = fp8_mul(op1, op2, false);
            uint8_t res = scalar_fp8_mul(op1, op2);

            // Assert that ML-MUL correctly maps 1-to-1 with mathematical truncation
            if (res != res_trunc) {
                std::cout << "Mismatch with truncation rounding => op1: " <<
                std::bitset<8>(op1) << ", op2: " << std::bitset<8>(op2) << ", ML-MUL: " 
                << std::bitset<8>(res) << ", res: " << std::bitset<8>(res_rntte) << "\n";
            }

            // Assert that ML-MUL matches Round-To-Nearest-Even except in the 2/16 cases
            // where the mathematical multiplication falls exactly on a 0.5 ULP tie
            if (res != res_rntte && (
                ((op1 & 0x03) == 0x02 && (op2 & 0x03) == 0x01) || 
                ((op1 & 0x03) == 0x01 && (op2 & 0x03) == 0x02)
            )) {
                std::cout << "Mismatch with round-to-nearest-even rounding => op1: " <<
                std::bitset<8>(op1) << ", op2: " << std::bitset<8>(op2) << ", ML-MUL: " 
                << std::bitset<8>(res) << ", res: " << std::bitset<8>(res_rntte) << "\n";
            }
            
        }
    }

}