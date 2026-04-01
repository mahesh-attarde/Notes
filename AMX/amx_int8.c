// COMPILE: icx -g amx_int8.c -o amx_int8 -O2 -march=sapphirerapids  -fno-strict-aliasing
// RUN: ./amx_int8

// SDE: sde64 -spr -debugtrace -- ./amx_int8
// XED: xed64 -i  ./amx_int8 | grep AMX
// DISASM: objdumpfunc ./amx_int8 int8_matmul
// HW: taskset -c 0 ./amx_int8

/*
cabbie amx_int8.c
cabbie -p DISASM  amx_int8.c
*/

#include <sys/prctl.h>
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1022
#endif

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

#define alignas(x) __attribute__((aligned(x)))

static int arch_prctl_req_tiledata(void) {
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (ret != 0) {
        fprintf(stderr, "ARCH_REQ_XCOMP_PERM for AMX tiledata failed: errno=%d\n", errno);
        return -1;
    }
    return 0;
}


static void tile_print(const int32_t *m, int rows, int cols, int ld) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%8d ", m[r * ld + c]);
    }
    printf("\n");
  }
}

// AMX Utilties
typedef struct __attribute__((packed, aligned(64))) {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];           // number of bytes per line in each TILEDATA file . Maximum of 64.
  uint8_t rows[16];             // number of rows in each TILEDATA . Maximum of 16.
} tilecfg_t;

#define CHECK_MEM_EQ(src,dst)  memcmp(src, dst, sizeof(src)) == 0

  // C = A x B
  // A: int8 matrix  (M x K)
  // B: uint8 matrix (K x N)
  // C: int32 matrix (M x N)
void int8_matmul(){
  const int M = 16;
  const int N = 16;
  const int K = 16;

  alignas(64)  int8_t SrcA[M*K];
  alignas(64) uint8_t SrcB[K*N];
  alignas(64) uint32_t DstC[M*N];
  tilecfg_t cfg;

  for (int i = 0; i < M * K; i++) 
    SrcA[i] = (int8_t)((i % 7) - 3);
  for (int i = 0; i < K * N; i++)
    SrcB[i] = (uint8_t)(i % 13);
  memset(DstC, 0, sizeof(DstC));


  memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;
  cfg.start_row = 0;

  cfg.rows[0] = M;
  cfg.colsb[0] = K;
  cfg.rows[1] = K;
  cfg.colsb[1] = N;
  cfg.rows[2] = M;
  cfg.colsb[2] = N;

  _tile_loadconfig(&cfg);
  _tile_zero(2);
  _tile_loadd(0, SrcA, 16);
  _tile_loadd(1, SrcB, 16);
   // tdpbusd: C += A(int8) * B(uint8) dot-products
  //_tile_dpbusd(2, 0, 1);
  _tile_stored(2, DstC, 16*4);

  _tile_release();
   tile_print((int32_t*)DstC, M, N, N);
}

int main(){
  if (arch_prctl_req_tiledata() != 0) {
    fprintf(stderr, " No AMX HW or OS Support!.\n");
  }
  int8_matmul();
    return 0;
}
