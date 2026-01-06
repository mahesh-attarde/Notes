## Notes on CPU Perf Investigations/tools/recipes

### Memory Profiling on CPU
+ Cache Specs 
   - Linux
   ```
    lscpu  # just size
    cat  /sys/devices/system/cpu/cpu0/cache/index  # all details
    lscpu -C
    NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL   SETS PHY-LINE COHERENCY-SIZE
    L1d       48K     5.6M   12 Data            1     64        1             64
    L1i       32K     3.8M    8 Instruction     1     64        1             64
    L2         2M     240M   16 Unified         2   2048        1             64
    L3     112.5M     225M   15 Unified         3 122880        1             64
   ```
   - Windows
    ```
    Read Settings, I did not have usecase for now.
    ```
+ [Perf Ninja Posts](https://easyperf.net/notes/)
+ [SDE Footprint](https://easyperf.net/blog/2024/02/12/Memory-Profiling-Part3) ( SDE LOG Explaination)
+ Memory Profilers heap_profier, memory_profiler, valgrind-cachegrind
+ Linux Perf
  - Hardware Events (https://perfmon-events.intel.com/)
  - Commands https://www.brendangregg.com/perf.html
  - ```
    # Various basic CPU statistics, system wide, for 10 seconds:
    perf stat -e cycles,instructions,cache-references,cache-misses,bus-cycles -a sleep 10
    # Various CPU level 1 data cache statistics for the specified command:
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores comman
    ```
  - Cache line sharing https://joemario.github.io/blog/2016/09/01/c2c-blog/
    ```
    https://github.com/joemario/perf-c2c-usage-files/blob/master/perf-c2c-usage.out
    ```
  - hardware Events CYCLE_ACTIVITY.STALLS_L1D_MISS, CYCLE_ACTIVITY.STALLS_L2_MISS
  - perf mem did not use yet
  - There is More on INTEL TMA https://github.com/andikleen/pmu-tools/tree/master
    ```
    ./pmu-tools/toplev.py --core S0-C0 -l3 -v --no-desc taskset -c 0 ./app.exe
    ```
+ Tuning Intel CPU
   - Optimization Manual Chapter  Optimizing Cache usage.
   - Chapter on Using Performance Monitering Events is super helpful, i wonder why it was placed last since for correct profiling we need it before other chapters(solutions)
