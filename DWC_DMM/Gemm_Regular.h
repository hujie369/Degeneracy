#ifndef GEMM_REGULAR_H
#define GEMM_REGULAR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define beta(i, j) B[(j) * ldB + (i)]  // map beta( i,j ) to array B
#define gamma(i, j) C[(j) * ldC + (i)] // map gamma( i,j ) to array C

#define min(x, y) ((x) < (y) ? x : y)

// 函数声明
void MyGemm(int, int, int, uint8_t *, int, int8_t *, int, int *, int);
void LoopFive(int, int, int, uint8_t *, int, int8_t *, int, int *, int, uint8_t *, int8_t *);
void LoopFour(int, int, int, uint8_t *, int, int8_t *, int, int *, int, uint8_t *, int8_t *);
void LoopThree(int, int, int, uint8_t *, int, int8_t *, int *, int, uint8_t *);
void LoopTwo(int, int, int, uint8_t *, int8_t *, int *, int);
void LoopOne(int, int, int, uint8_t *, int8_t *, int *, int);
void Gemm_MRxNRKernel_Packed(int, uint8_t *, int8_t *, int *, int);

#endif