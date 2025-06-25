#include "Gemm_Custom.h"
#include "Pack.h"


void MyGemm(int m, int n, int k,
            uint8_t *A, int ldA, EncodingType *B, int ldB, int *C, int ldC,
            int8_t *Codebook)
{
  if (MC % MR != 0 || NC % NR != 0)
  {
    printf("MC or NC must be multiples of MR or NR\n");
    exit(0);
  }

  // 申请Atilde内存, 要进行扩充, 因此缓存空间要乘上factor
  int16_t *Atilde = (int16_t *)malloc(MC * KC * factor * sizeof(int16_t));
  // 申请Btilde内存, 由于一个encoding对应一个参数向量, 因此缓存不需要太大
  uint8_t *Btilde = (uint8_t *)malloc(KC * NC * sizeof(uint8_t));

  LoopFive(m, n, k, A, ldA, B, ldB, C, ldC, Atilde, Btilde, Codebook);
  free(Atilde);
  free(Btilde);
}

void LoopFive(int m, int n, int k,
              uint8_t *A, int ldA, EncodingType *B, int ldB, int *C, int ldC,
              int16_t *Atilde, uint8_t *Btilde, int8_t *Codebook)
{
  for (int j = 0; j < n; j += NC)
  {
    int jb = min(NC, n - j); /* Last loop may not involve a full block */
    // Encoding中一个数对应长度为NR的参数向量, 分割时要注意
    int js = j / NR;
    LoopFour(m, jb, k, A, ldA, &beta(0, js), ldB, &gamma(0, j), ldC, Atilde, Btilde, Codebook);
  }
}

void LoopFour(int m, int n, int k,
              uint8_t *A, int ldA, EncodingType *B, int ldB, int *C, int ldC,
              int16_t *Atilde, uint8_t *Btilde, int8_t *Codebook)
{
  for (int p = 0; p < k; p += KC)
  {
    int pb = min(KC, k - p); /* Last loop may not involve a full block */

    // 原先的Btilde大小为KC x NC, 而Encoding的大小只有KC x (NC / NR)
    int ns = (n + NR - 1) / NR;
    // Pack B to Btilde, use elements in B(Encoding) to index uint8_t value from Codebook
    uint8_t *tmp;
    uint8_t *b = Btilde;
    for (int y = 0; y < ns; y++)
    {
      for (int x = 0; x < pb; x++)
      {
        tmp = Codebook + NR * B[y * ldB + p + x];
        for (int j = 0; j < NR; j++)
          *b++ = tmp[j];
      }
    }
    LoopThree(m, n, pb, &alpha(0, p), ldA, Btilde, C, ldC, Atilde);
  }
}

void LoopThree(int m, int n, int k,
               uint8_t *A, int ldA, uint8_t *Btilde, int *C, int ldC,
               int16_t *Atilde)
{
  for (int i = 0; i < m; i += MC)
  {
    int ib = min(MC, m - i); /* Last loop may not involve a full block */
    PackBlockA_MCxKC_DMM(ib, k, &alpha(i, 0), ldA, Atilde);
    LoopTwo(ib, n, k, Atilde, Btilde, &gamma(i, 0), ldC);
  }
}

void LoopTwo(int m, int n, int k, int16_t *Atilde, uint8_t *Btilde, int *C, int ldC)
{
  for (int j = 0; j < n; j += NR)
  {
    int jb = min(NR, n - j);
    LoopOne(m, jb, k, Atilde, &Btilde[j * k], &gamma(0, j), ldC);
  }
}

void LoopOne(int m, int n, int k, int16_t *Atilde, uint8_t *MicroPanelB, int *C, int ldC)
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
          *c++ = 0;
      }
      for (int y = n; y < NR; y++)
        for (int x = 0; x < MR; x++)
          *c++ = 0;
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
    Gemm_MRxNRKernel_Packed(k, &Atilde[i * k * factor], MicroPanelB, Ctilde, MR);
    // 将对应的值写回C
    for (int y = 0; y < n; y++)
    {
      for (int x = 0; x < ib; x++)
      {
        gamma(i + x, y) = Ctilde[y * MR + x];
      }
    }
  }
}

void Gemm_MRxNRKernel_Packed(int k, int16_t *MP_A, uint8_t *MP_B, int *Ctilde, int ldC)
{
  int gamma[MR * NR];   // scratchPad
  int16_t *alpha_p;
  int step = MR * factor;

  // 将Ctilde中数据加载到一起方便频繁读写
  int *c = Ctilde;
  int *g = gamma;
  for (int y = 0; y < NR; y++)
  {
    for (int x = 0; x < MR; x++)
    {
      *g++ = *c++;
    }
  }

  for (int p = 0; p < k; p++)
  {
    g = gamma;
    for (int j = 0; j < NR; j++)
    {
      alpha_p = MP_A + MR * MP_B[j];
      for (int i = 0; i < MR; i++)
      {
        *g++ += *alpha_p++;
      }
    }
    MP_A += step;
    MP_B += NR;
  }

  // 将scratchPad数据写回Ctilde
  g = gamma;
  c = Ctilde;
  for (int y = 0; y < NR; y++)
  {
    for (int x = 0; x < MR; x++)
    {
      *c++ = *g++;
    }
  }
}