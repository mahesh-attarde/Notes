//   COMPILE: g++ -O3 -march=sapphirerapids -fno-exceptions -fno-rtti vpdpbusd_bench_gnu.cpp -o bench
//   clang++ -O3 -march=sapphirerapids vpdpbusd_bench_gnu.cpp -o bench
//   RUN: taskset -c 2 ./bench

/*
cabbie vpdp.cpp


[COMPILE[0]] clang++ -O3 -march=sapphirerapids vpdp.cpp -o bench
[RUN[0]] taskset -c 2 ./bench
VPDPBUSD latency (cycles/inst):    3.036
VPDPBUSD latency unroll8 (cycles/inst):    3.130
VPDPBUSD throughput (cycles/inst): 0.758  (inst/cycle 1.320)
*/
#include <cstdint>
#include <cstdio>
#include <x86intrin.h>

static inline uint64_t tsc_start() {
  unsigned aux;
  _mm_mfence(); _mm_lfence();
  uint64_t t = __rdtscp(&aux);
  _mm_lfence();
  return t;
}
static inline uint64_t tsc_stop() {
  unsigned aux;
  _mm_lfence();
  uint64_t t = __rdtscp(&aux);
  _mm_lfence(); _mm_mfence();
  return t;
}

/*
vpdpbusd is accumulate into destination, zmm0 = zmm0 + dot_product(zmm1, zmm2)
zmm0 (Itereation 0) write blocks zmm0 (Itereation 1) read - TRUE RAW DEP.
That forces the CPU to wait for the result of one vpdpbusd before the next 
can complete, preventing overlap. In steady state, the loop throughput becomes 
limited by the instruction’s latency rather than execution bandwidth.
This can incur branching overhead
*/
static double lat_zmm(int iters) {
  uint64_t t0 = tsc_start();
  asm volatile(
    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
    "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
    "mov %[n], %%ecx\n\t"
    "1:\n\t"
    // GAS AT&T operand order: src2, src1, dst
    "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
    "dec %%ecx\n\t"
    "jnz 1b\n\t"
    :
    : [n] "r"(iters)
    : "ecx", "zmm0", "zmm1", "zmm2", "cc");
  uint64_t t1 = tsc_stop();
  return double(t1 - t0) / iters;
}

/*
Removing Looping/branching overhead by increasing depth of pipeline 
*/
 static double lat_zmm_unroll8(int outer_iters) {
   uint64_t t0 = tsc_start();
   asm volatile(
     "vpxord %%zmm0, %%zmm0, %%zmm0\n\t" // accumulator (dst)
     "vpxord %%zmm1, %%zmm1, %%zmm1\n\t" // src1
     "vpxord %%zmm2, %%zmm2, %%zmm2\n\t" // src2

     "mov %[n], %%ecx\n\t"
     "1:\n\t"
     // 8x dependent chain on the *same* dst (zmm0).
     // AT&T order: src2, src1, dst
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
     "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"

     "dec %%ecx\n\t"
     "jnz 1b\n\t"
     :
     : [n] "r"(outer_iters)
     : "ecx", "zmm0", "zmm1", "zmm2", "cc");
   uint64_t t1 = tsc_stop();

   constexpr int UNROLL = 8;
   return double(t1 - t0) / (double(outer_iters) * UNROLL); // cycles per vpdpbusd
 }


/*
To measure throughput, you instead use several different destination registers, 
each forming its own independent chain:
chain A: zmm0 = zmm0 + f(zmm1,zmm2)
chain B: zmm3 = zmm3 + f(zmm1,zmm2)
chain C: zmm4 = zmm4 + f(zmm1,zmm2)
source vectors the same (zmm1, zmm2) across all operations. That’s fine because
they are read-only in the loop and don’t create loop-carried dependencies.
Each individual accumulator still has a dependency chain (zmm0 depends on previous zmm0). 
If the instruction latency is, say, 4 cycles, then one chain can only issue one op every 4 cycles.
To reach peak throughput, you need enough independent chains so that while chain A is waiting, 
chains B/C/D… can execute.
#independent accumulators >= latency / reciprocal_throughput
You don’t know those numbers upfront, so you try 4, 6, 8, 12 accumulators and 
see where performance plateaus.
*/
static double tp_zmm(int iters) {
  uint64_t t0 = tsc_start();
  asm volatile(
    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
    "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
    "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
    "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
    "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
    "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
    "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
    "mov %[n], %%ecx\n\t"
    "1:\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm3\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm4\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm5\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm6\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm7\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm0\n\t"
    "vpdpbusd %%zmm2, %%zmm1, %%zmm3\n\t"
    "dec %%ecx\n\t"
    "jnz 1b\n\t"
    :
    : [n] "r"(iters)
    : "ecx", "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7", "cc");
  uint64_t t1 = tsc_stop();
  constexpr int insts_per_iter = 8;
  return double(t1 - t0) / (double(iters) * insts_per_iter);
}

int main() {
  const int it_lat = 20000000;
  const int it_tp  = 10000000;

  asm volatile("vzeroupper" ::: "memory");
  std::printf("VPDPBUSD latency (cycles/inst):    %.3f\n", lat_zmm(it_lat));

   double best = 1e100;
   for (int trial = 0; trial < 7; ++trial) {
     double cpi = lat_zmm_unroll8(it_lat);
     if (cpi < best) best = cpi;
   }
   std::printf("VPDPBUSD latency unroll8 (cycles/inst):    %.3f\n", best );

  std::printf("VPDPBUSD throughput (cycles/inst): %.3f  (inst/cycle %.3f)\n", tp, 1.0 / tp);
}
