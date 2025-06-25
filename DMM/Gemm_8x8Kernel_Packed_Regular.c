#include <stdio.h>
#include <stdint.h>


void Gemm_MRxNRKernel_Packed(int k, uint8_t *MP_A, int8_t *MP_B, int *Ctilde, int ldC)
{
  int gamma[MR * NR];
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
      for (int i = 0; i < MR; i++)
      {
        *g++ += MP_A[i] * MP_B[j];
      }
    }
    MP_A += MR;
    MP_B += NR;
  }

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

