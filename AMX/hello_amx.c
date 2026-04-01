// COMPILE: icx -g hello_amx.c -o hello_amx -O2 -march=sapphirerapids  -fno-strict-aliasing
// RUN: ./hello_amx

// SDE: sde64 -spr -debugtrace -- ./hello_amx
// XED: xed64 -i  ./hello_amx | grep AMX
// DISASM: objdumpfunc ./hello_amx hello_intrinsic

/*
cabbie hello_amx.c
cabbie -p DISASM  hello_amx.c
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

// AMX Utilties
typedef struct __attribute__((packed, aligned(64))) {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];           // number of bytes per line in each TILEDATA file . Maximum of 64.
  uint8_t rows[16];             // number of rows in each TILEDATA . Maximum of 16.
} tilecfg_t;

#define CHECK_MEM_EQ(src,dst)  memcmp(src, dst, sizeof(src)) == 0

#define ROWS 8
#define COLSB 8
#define STRIDE COLSB

void hello_intrinsic(){

  alignas(64) uint8_t SrcA[ROWS*COLSB];
  alignas(64) uint8_t SrcB[ROWS*COLSB];
  tilecfg_t cfg;
  
  for(int i = 0; i < ROWS*COLSB; i++){
    SrcA[i] = i;
  }

  memset(SrcB, 0, sizeof(SrcB));
  memset(&cfg, 0, sizeof(cfg));

  cfg.palette_id = 1;
  cfg.start_row = 0;
  cfg.colsb[0] = COLSB;
  cfg.rows[0] = ROWS;

  _tile_loadconfig(&cfg);
  _tile_loadd(0, SrcA, STRIDE);
  _tile_stored(0, SrcB, STRIDE);
  _tile_release();

  if(CHECK_MEM_EQ(SrcA, SrcB)){
    printf("AMX Intrinsic Test Passed!\n");
  } else {
    printf("AMX Intrinsic Test Failed!\n");
  }
}

int main(){
  if (arch_prctl_req_tiledata() != 0) {
    fprintf(stderr, " No AMX HW or OS Support!.\n");
  }
  hello_intrinsic();
  return 0;
}
