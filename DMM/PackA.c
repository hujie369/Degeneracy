#include <stdint.h>

#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define min(x, y) ((x) < (y) ? (x) : (y))


void PackMicroPanelA_MRxKC(int m, int k, uint8_t *A, int ldA, uint8_t *Atilde)
{
  if (m == MR) /* Full row size micro-panel.*/
  {
    for (int p = 0; p < k; p++)
      for (int i = 0; i < MR; i++)
        *Atilde++ = alpha(i, p);
  }
  else /* Not a full row size micro-panel.  To be implemented */
  {
    for (int p = 0; p < k; p++)
    {
      for (int i = 0; i < m; i++)
        *Atilde++ = alpha(i, p);
      for (int i = m; i < MR; i++)
        *Atilde++ = 0;
    }
  }
}

void PackBlockA_MCxKC(int m, int k, uint8_t *A, int ldA, uint8_t *Atilde)
{
  for (int i = 0; i < m; i += MR)
  {
    int ib = min(MR, m - i);
    PackMicroPanelA_MRxKC(ib, k, &alpha(i, 0), ldA, Atilde);
    Atilde += ib * k;
  }
}


// quant实现
void PackMicroPanelA_MRxKC_quant(int m, int k, uint8_t *A, int ldA, short *Atilde)
{
  int8_t low = -factor / 2;
  int8_t high = factor / 2;
  if (m == MR) /* Full row size micro-panel.*/
  {
    for (int p = 0; p < k; p++)
    {
      for (int8_t q = low; q < high; q++)
        for (int i = 0; i < MR; i++)
          *Atilde++ = alpha(i, p) * q;
    }
  }
  else
  {
    for (int p = 0; p < k; p++)
    {
      for (int8_t q = low; q < high; q++)
      {
        for (int i = 0; i < m; i++)
          *Atilde++ = alpha(i, p) * q;
        for (int i = m; i < MR; i++)
          *Atilde++ = 0;
      }
    }
  }
}

void PackBlockA_MCxKC_quant(int m, int k, uint8_t *A, int ldA, short *Atilde)
{
  for (int i = 0; i < m; i += MR)
  {
    int ib = min(MR, m - i);
    PackMicroPanelA_MRxKC_quant(ib, k, &alpha(i, 0), ldA, Atilde);
    Atilde += (ib * k * factor);
  }
}
