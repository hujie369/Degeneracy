#ifndef GEMM_CUSTOM_H
#define GEMM_CUSTOM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "Utils.h"

#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define beta(i, j) B[(j) * ldB + (i)]  // map beta( i,j ) to array B
#define gamma(i, j) C[(j) * ldC + (i)] // map gamma( i,j ) to array C
#define min(x, y) ((x) < (y) ? x : y)

// typedef uint32_t EncodingType;

// 根据宏选择变量的类型
#ifdef TYPE_IS_UINT8
    typedef uint8_t EncodingType;
#elif TYPE_IS_UINT16
    typedef uint16_t EncodingType;
#elif TYPE_IS_UINT32
    typedef uint32_t EncodingType;
#else
    #error "Invalid TYPE_ID macro"
#endif

// 函数声明
void MyGemm(int, int, int, uint8_t *, int, EncodingType *, int, int *, int, int8_t *);
void LoopFive(int, int, int, uint8_t *, int, EncodingType *, int, int *, int, int16_t *, uint8_t *, int8_t *);
void LoopFour(int, int, int, uint8_t *, int, EncodingType *, int, int *, int, int16_t *, uint8_t *, int8_t *);
void LoopThree(int, int, int, uint8_t *, int, uint8_t *, int *, int, int16_t *);
void LoopTwo(int, int, int, int16_t *, uint8_t *, int *, int);
void LoopOne(int, int, int, int16_t *, uint8_t *, int *, int);
void Gemm_MRxNRKernel_Packed(int, int16_t *, uint8_t *, int *, int);

// 根据codebook的大小确定encoding的类型

#endif