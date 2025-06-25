#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// 根据不同平台包含不同的头文件
#if defined(__APPLE__) || defined(__MACH__)
#include <AvailabilityMacros.h>
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define dabs(x) ((x) < 0 ? -(x) : x)
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
double FLA_Clock_helper(void);
double FLA_Clock(void);

void RandomMatrix_uint8(uint8_t *A, size_t size, int quant_min, int quant_max);
void Load_int8(int8_t *A, size_t size, const char *path);
void Load_shape(int *A, size_t size, const char *path);
void Load_encoding(EncodingType *A, size_t size, const char *path);
void int_to_double(double *d_arr, int *i_arr, size_t size);
void uint8_to_double(double *d_arr, uint8_t *i_arr, size_t size);
void int8_to_double(double *d_arr, int8_t *i_arr, size_t size);
double MaxAbsDiff(int m, int n, double *ap, int lda, double *bp, int ldb);


// 全局变量声明
extern double gtod_ref_time_sec;


#endif