#include "Pack.h"


void PackMicroPanelB_KCxNR(int k, int n, int8_t *B, int ldB, int8_t *Btilde)
{
  if (n == NR) /* Full column width micro-panel.*/
  {
    for (int p = 0; p < k; p++)
      for (int j = 0; j < NR; j++)
        *Btilde++ = beta(p, j);
  }
  else /* Not a full row size micro-panel.  We pad with zeroes. */
  {
    for (int p = 0; p < k; p++)
    {
      for (int j = 0; j < n; j++)
        *Btilde++ = beta(p, j);
      for (int j = n; j < NR; j++)
        *Btilde++ = 0;
    }
  }
}

void PackPanelB_KCxNC(int k, int n, int8_t *B, int ldB, int8_t *Btilde)
{
  for (int j = 0; j < n; j += NR)
  {
    int jb = min(NR, n - j);
    PackMicroPanelB_KCxNR(k, jb, &beta(0, j), ldB, Btilde);
    Btilde += k * jb;
  }
}

/* Pack A */
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

// DMM实现
void PackMicroPanelA_MRxKC_DMM(int m, int k, uint8_t *A, int ldA, int16_t *Atilde)
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

void PackBlockA_MCxKC_DMM(int m, int k, uint8_t *A, int ldA, int16_t *Atilde)
{
  for (int i = 0; i < m; i += MR)
  {
    int ib = min(MR, m - i);
    PackMicroPanelA_MRxKC_DMM(ib, k, &alpha(i, 0), ldA, Atilde);
    Atilde += (ib * k * factor);
  }
}