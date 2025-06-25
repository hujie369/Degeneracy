#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define beta(i, j) B[(j) * ldB + (i)]  // map beta( i,j ) to array B
#define gamma(i, j) C[(j) * ldC + (i)] // map gamma( i,j ) to array C

#define min(x, y) ((x) < (y) ? x : y)

// int MC, KC;

void LoopFive(int, int, int, uint8_t *, int, int8_t *, int, int *, int, uint8_t *, int8_t *);
void LoopFour(int, int, int, uint8_t *, int, int8_t *, int, int *, int, uint8_t *, int8_t *);
void LoopThree(int, int, int, uint8_t *, int, int8_t *, int *, int, uint8_t *);
void LoopTwo(int, int, int, uint8_t *, int8_t *, int *, int);
void LoopOne(int, int, int, uint8_t *, int8_t *, int *, int);
void Gemm_MRxNRKernel_Packed(int, uint8_t *, int8_t *, int *, int);
void PackBlockA_MCxKC(int, int, uint8_t *, int, uint8_t *);
void PackPanelB_KCxNC(int, int, int8_t *, int, int8_t *);


void MyGemm(int m, int n, int k, uint8_t *A, int ldA,
            int8_t *B, int ldB, int *C, int ldC)
{
  // 分配Btilde内存和Atilde内存
  uint8_t *Atilde = (uint8_t *)malloc(MC * KC * sizeof(uint8_t));
  int8_t *Btilde = (int8_t *)malloc(KC * NC * sizeof(int8_t));

  LoopFive(m, n, k, A, ldA, B, ldB, C, ldC, Atilde, Btilde);

  free(Atilde);
  free(Btilde);
}

void LoopFive(int m, int n, int k,
  uint8_t *A, int ldA, int8_t *B, int ldB, int *C, int ldC,
  uint8_t *Atilde, int8_t *Btilde)
{
  for (int j = 0; j < n; j += NC)
  {
    int jb = min(NC, n - j); /* Last loop may not involve a full block */
    LoopFour(m, jb, k, A, ldA, &beta(0, j), ldB, &gamma(0, j), ldC, Atilde, Btilde);
  }
}

void LoopFour(int m, int n, int k,
  uint8_t *A, int ldA, int8_t *B, int ldB, int *C, int ldC,
  uint8_t *Atilde, int8_t *Btilde)
{
  for (int p = 0; p < k; p += KC)
  {
    int pb = min(KC, k - p); /* Last loop may not involve a full block */
    PackPanelB_KCxNC(pb, n, &beta(p, 0), ldB, Btilde);
    LoopThree(m, n, pb, &alpha(0, p), ldA, Btilde, C, ldC, Atilde);
  }
}

void LoopThree(int m, int n, int k,
  uint8_t *A, int ldA, int8_t *Btilde, int *C, int ldC,
  uint8_t *Atilde)
{
  for (int i = 0; i < m; i += MC)
  {
    int ib = min(MC, m - i); /* Last loop may not involve a full block */
    PackBlockA_MCxKC(ib, k, &alpha(i, 0), ldA, Atilde);
    LoopTwo(ib, n, k, Atilde, Btilde, &gamma(i, 0), ldC);
  }
}

void LoopTwo(int m, int n, int k, uint8_t *Atilde, int8_t *Btilde, int *C, int ldC)
{
  for (int j = 0; j < n; j += NR)
  {
    int jb = min(NR, n - j);
    LoopOne(m, jb, k, Atilde, &Btilde[j * k], &gamma(0, j), ldC);
  }
}


void LoopOne(int m, int n, int k, uint8_t *Atilde, int8_t *MicroPanelB, int *C, int ldC)
{
  int Ctilde[MR * NR];
  for (int i = 0; i < m; i += MR)
  {
    int ib = min(MR, m - i);
    int *c = Ctilde;
    // 使用Ctilde进行补0
    if (n < NR || ib < MR)
    {
      for (int y = 0; y < n; y++)
      {
        for (int x = 0; x < ib; x++)
          *c++ = gamma(i + x, y);
        for (int x = ib; x < MR; x++)
          *c++ = 0.0;
      }
      for (int y = n; y < NR; y++)
        for (int x = 0; x < MR; x++)
          *c++ = 0.0;
    }
    else
    {
      for (int y = 0; y < NR; y++)
      {
        for (int x = 0; x < MR; x++)
        {
          *c++ = gamma(i + x, y);
        }
      }
    }
 
    // 使用Ctilde进行计算
    Gemm_MRxNRKernel_Packed(k, &Atilde[i * k], MicroPanelB, Ctilde, MR);

    // 将计算好的值写回C, 注意Ctilde不一定是全满的
    c = Ctilde;
    for (int y = 0; y < n; y++)
    {
      for (int x = 0; x < ib; x++)
      {
        gamma(i + x, y) = *c++;
      }
    }
  }
}

