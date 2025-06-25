#ifndef PACK_H
#define PACK_H

#include <stdint.h>

#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define beta(i, j) B[(j) * ldB + (i)]  // map beta( i,j ) to array B
#define min(x, y) ((x) < (y) ? x : y)

void PackMicroPanelB_KCxNR(int k, int n, int8_t *B, int ldB, int8_t *Btilde);
void PackPanelB_KCxNC(int k, int n, int8_t *B, int ldB, int8_t *Btilde);
void PackMicroPanelA_MRxKC(int m, int k, uint8_t *A, int ldA, uint8_t *Atilde);
void PackBlockA_MCxKC(int m, int k, uint8_t *A, int ldA, uint8_t *Atilde);
void PackMicroPanelA_MRxKC_DMM(int m, int k, uint8_t *A, int ldA, int16_t *Atilde);
void PackBlockA_MCxKC_DMM(int m, int k, uint8_t *A, int ldA, int16_t *Atilde);

#endif