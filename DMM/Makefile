#HOME      := /Users/rvdg
# Make sure you have BLIS installed in your home directory (or some other BLAS library)
BLAS_LIB  := $(HOME)/blis/lib/libblis.a
BLAS_INC  := $(HOME)/blis/include/blis

# indicate how the object files are to be created
CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O3 -I$(BLAS_INC) -m64 -mavx2 -std=c99 -march=native -fopenmp -D_POSIX_C_SOURCE=200809L
FFLAGS     := $(CFLAGS) 

# set the range of experiments to be performed
NREPEATS   := 5 #       number of times each experiment is repeated.  The best time is reported.
NFIRST     := 64#     smallest size to be timed.
NLAST_SMALL:= 500#    largest size to be timed for slow implementations.
NLAST_SMALLISH:= 1000#    largest size to be timed for slowish implementations.
NLAST      := 2000#   largest size to be timed for fast implementations.
NINC       := 64#     increment between sizes.

LDFLAGS    := -lpthread -m64 -lm -fopenmp

UTIL_OBJS  := FLA_Clock.o MaxAbsDiff.o RandomMatrix.o

# -------------- Set blocking parameters ----------

MC :=  32
KC :=  480
NC := 2048
# factor := 8

# ---------------------  
SRC_Five_Loops_Packed_8x8Kernel_Regular := driver.c Gemm_Five_Loops_Packed_MRxNRKernel_Regular.c Gemm_8x8Kernel_Packed_Regular.c PackA.c PackB.c

Five_Loops_Packed_8x8Kernel_Regular:  $(SRC_Five_Loops_Packed_8x8Kernel_Regular) $(UTIL_OBJS) Makefile
	$(LINKER) $(CFLAGS) -D'factor=16' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)' -D'MR=8' -D'NR=8' $(SRC_Five_Loops_Packed_8x8Kernel_Regular) $(UTIL_OBJS) $(BLAS_LIB) -o driver_Five_Loops_Packed_8x8Kernel_Regular.x $(LDFLAGS) 
	echo "$(NREPEATS) $(NFIRST) $(NLAST) $(NINC)" | ./driver_Five_Loops_Packed_8x8Kernel_Regular.x > data/output_Five_Loops_Packed_8x8Kernel_Regular.m 
	tail data/output_Five_Loops_Packed_8x8Kernel_Regular.m

# ---------------------  
SRC_Five_Loops_Packed_8x8Kernel_DMM := driver_DMM.c Gemm_Five_Loops_Packed_MRxNRKernel_DMM.c Gemm_8x8Kernel_Packed_DMM.c PackA.c PackB.c

Five_Loops_Packed_8x8Kernel_DMM_w2:  $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) Makefile
	$(LINKER) $(CFLAGS) -D'factor=4' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)'  -D'MR=8' -D'NR=8' $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) $(BLAS_LIB) -o driver_Five_Loops_Packed_8x8Kernel_DMM_w2.x $(LDFLAGS) 
	echo "$(NREPEATS) $(NFIRST) $(NLAST) $(NINC)" | ./driver_Five_Loops_Packed_8x8Kernel_DMM_w2.x > data/output_Five_Loops_Packed_8x8Kernel_DMM_w2.m
	tail data/output_Five_Loops_Packed_8x8Kernel_DMM_w2.m
# ---------------------  

Five_Loops_Packed_8x8Kernel_DMM_w3:  $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) Makefile
	$(LINKER) $(CFLAGS) -D'factor=8' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)'  -D'MR=8' -D'NR=8' $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) $(BLAS_LIB) -o driver_Five_Loops_Packed_8x8Kernel_DMM_w3.x $(LDFLAGS) 
	echo "$(NREPEATS) $(NFIRST) $(NLAST) $(NINC)" | ./driver_Five_Loops_Packed_8x8Kernel_DMM_w3.x > data/output_Five_Loops_Packed_8x8Kernel_DMM_w3.m
	tail data/output_Five_Loops_Packed_8x8Kernel_DMM_w3.m
# ---------------------  
SRC_Five_Loops_Packed_8x8Kernel_quant_ws := driver_quant.c Gemm_Five_Loops_Packed_MRxNRKernel_quant.c Gemm_8x8Kernel_Packed_quant_without_SIMD.c PackA.c PackB.c

Five_Loops_Packed_8x8Kernel_DMM_w4:  $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) Makefile
	$(LINKER) $(CFLAGS) -D'factor=16' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)'  -D'MR=8' -D'NR=8' $(SRC_Five_Loops_Packed_8x8Kernel_DMM) $(UTIL_OBJS) $(BLAS_LIB) -o driver_Five_Loops_Packed_8x8Kernel_DMM_w4.x $(LDFLAGS) 
	echo "$(NREPEATS) $(NFIRST) $(NLAST) $(NINC)" | ./driver_Five_Loops_Packed_8x8Kernel_DMM_w4.x > data/output_Five_Loops_Packed_8x8Kernel_DMM_w4.m
	tail data/output_Five_Loops_Packed_8x8Kernel_DMM_w4.m

# --------------------- 

clean:
	rm -f *.o *~ core *.x *.pdf
