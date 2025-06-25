#include <stdint.h>

#define beta(i, j) B[(j) * ldB + (i)] // map beta( i,j ) to array B
#define min(x, y) ((x) < (y) ? x : y)


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
