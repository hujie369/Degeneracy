#include "Utils.h"

void RandomMatrix_uint8(uint8_t *A, size_t size, int quant_min, int quant_max)
{
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < size; i++)
  {
    A[i] = quant_min + rand() % (quant_max - quant_min + 1);
  }
}

void Load_int8(int8_t *A, size_t size, const char *path)
{
  FILE *file = fopen(path, "rb");
  if (file == NULL)
  {
    // 使用 fprintf 输出包含错误路径和错误描述的信息
    fprintf(stderr, "Failed to open weight file: %s, Error: %s\n", path, strerror(errno));
    return;
  }
  size_t elements_read = fread(A, sizeof(int8_t), size, file);
  if (elements_read != size)
  {
    // 处理读取失败的情况
    fprintf(stderr, "Error: Failed to read the expected number of int8.\n");
    exit(EXIT_FAILURE);
  }
  fclose(file);
}

void Load_shape(int *A, size_t size, const char *path)
{
  FILE *file = fopen(path, "rb");
  if (file == NULL)
  {
    // 使用 fprintf 输出包含错误路径和错误描述的信息
    fprintf(stderr, "Failed to open weight file: %s, Error: %s\n", path, strerror(errno));
    return;
  }
  size_t elements_read = fread(A, sizeof(int), size, file);
  if (elements_read != size)
  {
    // 处理读取失败的情况
    fprintf(stderr, "Error: Failed to read the expected number of shape.\n");
    exit(EXIT_FAILURE);
  }
  fclose(file);
}

void Load_encoding(EncodingType *A, size_t size, const char *path)
{
  FILE *file = fopen(path, "rb");
  if (file == NULL)
  {
    // 使用 fprintf 输出包含错误路径和错误描述的信息
    fprintf(stderr, "Failed to open weight file: %s, Error: %s\n", path, strerror(errno));
    return;
  }
  size_t elements_read = fread(A, sizeof(EncodingType), size, file);
  if (elements_read != size)
  {
    // 处理读取失败的情况
    fprintf(stderr, "Error: Failed to read the expected number of encoding.\n");
    exit(EXIT_FAILURE);
  }
  fclose(file);
}

void int_to_double(double *d_arr, int *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (double)i_arr[i];
}

void uint8_to_double(double *d_arr, uint8_t *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (double)i_arr[i];
}

void int8_to_double(double *d_arr, int8_t *i_arr, size_t size)
{
  for (size_t i = 0; i < size; i++)
    d_arr[i] = (double)i_arr[i];
}


double MaxAbsDiff(int m, int n, double *ap, int lda, double *bp, int ldb)
/*
   MaxAbsDiff returns the maximum absolute difference over
   corresponding elements of matrices A and B.
*/
{
  double diff = 0.0;
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      if (dabs(ap[j * lda + i] - bp[j * lda + i]) > diff)
        diff = dabs(ap[j * lda + i] - bp[j * lda + i]);

  return diff;
}

/* FLA_Clock */
double gtod_ref_time_sec = 0.0;

// FLA_Clock 函数实现
double FLA_Clock(void)
{
  return FLA_Clock_helper();
}

// 根据不同平台实现 FLA_Clock_helper 函数
#if defined(__APPLE__) || defined(__MACH__)

double FLA_Clock_helper()
{
  mach_timebase_info_data_t timebase;
  mach_timebase_info(&timebase);

  uint64_t nsec = mach_absolute_time();

  double the_time = (double)nsec * 1.0e-9 * timebase.numer / timebase.denom;

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = the_time;

  return the_time - gtod_ref_time_sec;
}

#else

double FLA_Clock_helper()
{
  double the_time, norm_sec;
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)ts.tv_sec;

  norm_sec = (double)ts.tv_sec - gtod_ref_time_sec;

  the_time = norm_sec + ts.tv_nsec * 1.0e-9;

  return the_time;
}

#endif
