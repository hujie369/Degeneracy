#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define A(i, j) *(ap + (j) * lda + (i)) // map A( i,j )    to array ap    in column-major order

void RandomMatrix(int m, int n, double *ap, int lda)
/*
   RandomMatrix overwrite A with random values.
*/
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A(i, j) = drand48();
}


void RandomMatrix_int(int *A, size_t size, int quant_min, int quant_max, int rate)
{
  int tmp;
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < size; i++)
  {
    if (rand() % 100 < rate)
    {
      A[i] = 0;
    }
    else
    {
      do
      {
        tmp = quant_min + rand() % (quant_max - quant_min + 1);
      } while (tmp == 0);
      A[i] = tmp;
    }
  }
}


void int_to_double(double *d_arr, int *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (double)i_arr[i];
}


void int_to_s8(int8_t *d_arr, int *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (int8_t)i_arr[i];
}

void int_to_u8(uint8_t *d_arr, int *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (uint8_t)i_arr[i];
}