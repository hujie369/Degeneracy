#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Utils.h"
#include "Gemm_Custom.h"

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
      nrepeats,
      shape[3];

  double
      d_one = 1.0,
      dtime, dtime_best,
      diff, maxdiff = 0.0, gflops;

  int
      *C, *Cold;

  double
      *Ad,
      *Bd, *Cref, *Cres;

  // the path of weight file
  const char *shape_path = FILE_PATH(shape.bin);
  const char *codebook_path = FILE_PATH(codebook.bin);
  const char *encoding_path = FILE_PATH(encoding.bin);
  const char *weight_path = FILE_PATH(weights.bin);
  fflush(stdout);

  uint8_t *Au;
  int8_t *Weights;
  int8_t *Codebook;
  EncodingType *Encoding;

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
  Cres = (double *)malloc(ldC * n * sizeof(double));

  Ad = (double *)malloc(ldA * k * sizeof(double));
  Bd = (double *)malloc(ldB * n * sizeof(double));
  Cref = (double *)malloc(ldC * n * sizeof(double));

  Au = (uint8_t *)malloc(ldA * k * sizeof(uint8_t));
  /* Generate random matrix A, 8bit activation */
  RandomMatrix_uint8(Au, ldA * k, 0, 15);
  /* Load low-bit quantized weights */
  Weights = (int8_t *)malloc(k * n * sizeof(int8_t));
  Load_int8(Weights, k * n, weight_path);

  /* load Codebook and Encoding */
  Load_shape(shape, 3, shape_path);
  Codebook = (int8_t *)malloc(shape[0] * sizeof(int8_t));
  Encoding = (EncodingType *)malloc(shape[1] * shape[2] * sizeof(EncodingType));
  Load_int8(Codebook, shape[0], codebook_path);
  Load_encoding(Encoding, shape[1] * shape[2], encoding_path);

  // 对Codebook矩阵进行偏移, 以便索引
  int8_t offset = (int8_t)(factor / 2);
  for (int i = 0; i < shape[0]; i++)
  {
    Codebook[i] += offset;
  }

  /* Initilize matrix Cold */
  for (int i = 0; i < ldC * n; i++)
  {
    Cold[i] = 0;
  }

  uint8_to_double(Ad, Au, ldA * k);
  int8_to_double(Bd, Weights, ldB * n);

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
  printf(" %5d %8.4le %8.4le   ", n, dtime_best, gflops / dtime_best);
  fflush(stdout);


  /* Time MyGemm */
  for (irep = 0; irep < nrepeats; irep++)
  {
    /* Copy vector Cold to C */
    memcpy(C, Cold, ldC * n * sizeof(int));

    /* start clock */
    dtime = FLA_Clock();

    /* Compute C = A @ B + C */
    MyGemm(m, n, k, Au, ldA, Encoding, ldB, C, ldC, Codebook);

    /* stop clock */
    dtime = FLA_Clock() - dtime;

    if (irep == 0)
      dtime_best = dtime;
    else
      dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  }

  // 将计算结果C转移到double矩阵Cres中
  int_to_double(Cres, C, ldC * n);
  diff = MaxAbsDiff(m, n, Cres, ldC, Cref, ldC);
  maxdiff = (diff > maxdiff ? diff : maxdiff);

  printf(" %8.4le %8.4le %8.4le\n", dtime_best, gflops / dtime_best, diff);
  fflush(stdout);
  // printf("%.4f,", gflops / dtime_best);

  /* free the buffers */
  free(C);
  free(Cres);
  free(Ad);
  free(Bd);
  free(Cold);
  free(Cref);
  free(Au);
  free(Weights);
  free(Codebook);
  free(Encoding);

  // printf("];\n\n");
  // printf("%% Maximum difference between reference and your implementation: %le.\n", maxdiff);

  exit(0);
}
