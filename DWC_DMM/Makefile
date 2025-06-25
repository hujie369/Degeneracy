#HOME      := /Users/rvdg
# Make sure you have BLIS installed in your home directory (or some other BLAS library)
BLAS_LIB  := $(HOME)/blis/lib/libblis.a
BLAS_INC  := $(HOME)/blis/include/blis

# indicate how the object files are to be created
CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O3 -I$(BLAS_INC) -m64 -std=c99 -D_POSIX_C_SOURCE=200809L

# set the range of experiments to be performed
NREPEATS   := 20 #       number of times each experiment is repeated.  The best time is reported.
m          := 512
n          := 1000
# vit'k: 768, res18'k: 512
k          := 512

LDFLAGS    := -lpthread -m64 -lm -fopenmp
# -------------- Set blocking parameters ----------
# Gemm函数对A, B, C的分割大小, MC要被MR整除, NC要被NR整除
MC :=  32
KC :=  512
NC := 2000
# kernel函数对mirco C的分割大小, NR对应权重向量的长度
MR := 8
NR := 8
# 4, 8, 16 for 2, 3, 4 bit quantization
bit := 2
factor := 4

# typeId = 1, 2, 2 when bit = 2, 3, 4 for resnet18

typeId = 1
ifeq ($(typeId),0)
	CFLAGS += -DTYPE_IS_UINT8
else ifeq ($(typeId),1)
	CFLAGS += -DTYPE_IS_UINT16
else ifeq ($(typeId),2)
	CFLAGS += -DTYPE_IS_UINT32
else
$(error typeId 的值不合法: $(typeId))
endif

# ---------------------  
SRC_Regular_Method := Gemm_Regular.c Pack.c Utils.c driver_regular.c
Output := resnet18
# 加载模型权重的目录
folder := "./matrix_data/$(Output)/$(bit)_$(strip $(NR))/"

Regular_Method:  $(SRC_Regular_Method) Makefile
	$(LINKER) $(CFLAGS) -D'FOLDER=$(folder)' -D'factor=16' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)' -D'MR=$(MR)' -D'NR=$(NR)' $(SRC_Regular_Method) $(BLAS_LIB) -o driver_regular_method.x $(LDFLAGS)
	echo "$(NREPEATS) $(m) $(n) $(k)" | ./driver_regular_method.x >> data/$(Output)_regular_$(bit).txt
	tail data/$(Output)_regular_$(bit).txt

# ---------------------  
SRC_Custom_Method := driver_custom.c Gemm_Custom.c Pack.c Utils.c

Custom_Method:  $(SRC_Custom_Method) Makefile
	$(LINKER) $(CFLAGS) -D'FOLDER=$(folder)' -D'factor=$(factor)' -D'MC=$(MC)' -D'NC=$(NC)' -D'KC=$(KC)' -D'MR=$(MR)' -D'NR=$(NR)' $(SRC_Custom_Method) $(BLAS_LIB) -o driver_custom_method.x $(LDFLAGS)
	echo "$(NREPEATS) $(m) $(n) $(k)" | ./driver_custom_method.x >> data/$(Output)_custom_$(bit).txt
	tail data/$(Output)_custom_$(bit).txt

# ---------------------  

clean:
	rm -f *.o *~ core *.x *.pdf
