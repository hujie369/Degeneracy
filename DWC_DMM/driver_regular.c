#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Utils.h"
#include "Gemm_Regular.h"

#define FILE_PATH(x) FOLDER #x

/* Prototype for BLAS matrix-matrix multiplication routine (which we will
   use for the reference implementation */
void dgemm_(char *, char *,             // transA, transB
            int *, int *, int *,        // m, n, k
            double *, double *, int *,  // alpha, A, ldA
            double *, int *,            //        B, ldB
            double *, double *, int *); // beta,  C, ldC


int main(int argc, char *argv[])
{
  int
      m,
      n, k,
      ldA, ldB, ldC,
      irep,
      nrepeats;

  double
      d_one = 1.0,
      dtime, dtime_best,
      diff, maxdiff = 0.0, gflops;

  int
      *C, *Cold;

  double
      *Ad,
      *Bd, *Cref, *Ctmp;

  // the path of weight file 
  const char *weight_path = FILE_PATH(weights.bin);

  uint8_t *Au;
  int8_t *Bs;

  int quant_min = -factor / 2;
  int quant_max = factor / 2 - 1;

  /* Every time trial is repeated "repeat" times and the fastest run in recorded */
  // printf("%% number of repeats:");
  if (scanf("%d", &nrepeats) == EOF) {
    printf("scanf error");
    return 1;
  }
  // printf("%% %d\n", nrepeats);

  // printf("%% enter first, last, inc:");
  // scanf("%d%d%d", &first, &last, &inc);
  // printf("%% enter m, n, k: ");
  if (scanf("%d%d%d", &m, &n, &k) == EOF)
  {
    printf("scanf error");
    return 1;
  }

  // printf("data = [\n");
  // printf("%%  n          reference      |         current implementation \n");
  // printf("%%        time       GFLOPS   |    time       GFLOPS     diff \n");

  
  ldA = m;
  ldB = k;
  ldC = m;

  /* Gflops performed */
  gflops = 2.0 * m * n * k * 1e-09;

  C = (int *)malloc(ldC * n * sizeof(int));
  Cold = (int *)malloc(ldC * n * sizeof(int));
  Ctmp = (double *)malloc(ldC * n * sizeof(double));

  Ad = (double *)malloc(ldA * k * sizeof(double));
  Bd = (double *)malloc(ldB * n * sizeof(double));
  Cref = (double *)malloc(ldC * n * sizeof(double));

  Au = (uint8_t *)malloc(ldA * k * sizeof(uint8_t));
  Bs = (int8_t *)malloc(ldB * n * sizeof(int8_t));

  /* Generate random matrix A, 8bit activation */
  RandomMatrix_uint8(Au, ldA * k, 0, 15);

  /* load low-bit quantized weights */
  Load_int8(Bs, ldB * n, weight_path);

  /* Initilize matrix Cold */
  for (int i = 0; i < ldC * n; i++)
  {
    Cold[i] = 0;
  }

  uint8_to_double(Ad, Au, ldA * k);
  int8_to_double(Bd, Bs, ldB * n);

  for (irep = 0; irep < nrepeats; irep++)
  {
    /* Copy matrix Cold to Cref */
    int_to_double(Cref, Cold, ldC * n);

    /* start clock */
    dtime = FLA_Clock();

    /* Compute Cref = A B + Cref */
    dgemm_("No transpose", "No transpose",
            &m, &n, &k,
            &d_one, Ad, &ldA,
            Bd, &ldB,
            &d_one, Cref, &ldC);

    /* stop clock */
    dtime = FLA_Clock() - dtime;

    /* record the best time so far */
    if (irep == 0)
      dtime_best = dtime;
    else
      dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  }
  // printf(" %5d %8.4le %8.4le   ", n, dtime_best, gflops / dtime_best);
  fflush(stdout);


  /* Time MyGemm */
  for (irep = 0; irep < nrepeats; irep++)
  {
    /* Copy vector Cold to C */
    memcpy(C, Cold, ldC * n * sizeof(int));

    /* start clock */
    dtime = FLA_Clock();

    /* Compute C = A B + C */
    MyGemm(m, n, k, Au, ldA, Bs, ldB, C, ldC);

    /* stop clock */
    dtime = FLA_Clock() - dtime;

    if (irep == 0)
      dtime_best = dtime;
    else
      dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  }

  // 将计算结果C转移到double矩阵Ctmp中
  int_to_double(Ctmp, C, ldC * n);
  diff = MaxAbsDiff(m, n, Ctmp, ldC, Cref, ldC);
  maxdiff = (diff > maxdiff ? diff : maxdiff);

  // printf(" %8.4le %8.4le %8.4le\n", dtime_best, gflops / dtime_best, diff);
  // fflush(stdout);
  printf("%.4f,", gflops / dtime_best);

  /* free the buffers */
  free(C);
  free(Ctmp);
  free(Ad);
  free(Bd);
  free(Cold);
  free(Cref);
  free(Au);
  free(Bs);

  // printf("];\n\n");
  // printf("%% Maximum difference between reference and your implementation: %le.\n", maxdiff);

  exit(0);
}
