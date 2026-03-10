### Attempt to use VPMADD52L/VPMULUDQ instead of VPMULLQ on slow VPMULLQ targets
https://github.com/llvm/llvm-project/pull/171760

## Why the same logical operation can have different latency Conceptually both implementations compute the same mathematical result (a 64‑bit product), but:
+ VPMULLQ is a monolithic 64‑bit integer multiply, which is costly to implement in hardware at high frequency.
+ VPMADD52LUQ and VPMULUDQ operate on smaller chunks (52 or 32 bits), which are cheaper to multiply quickly. The ISA then lets software build up a full‑precision result from these cheaper building blocks.
That hardware design choice is why microarchitectural latency differs.

1. Hardware cost vs. “mathematical operation”
Mathematically:
VPMULLQ computes a full 64×64 → 64 product per lane.
A sequence using VPMULUDQ or VPMADD52LUQ plus adds/shifts computes the same product, but in stages.
From the ISA point of view, both are “64‑bit multiply”. But at the hardware level:

+ A full‑width 64‑bit multiplier is:
  Larger
  Slower to propagate carries
  More power‑hungry

+ A narrower multiplier (32‑bit or 52‑bit) is:
  Smaller
  Faster per operation
  Easier to pipeline
  So vendors often choose:

+ Implement “wide” operations (like 64‑bit vector mul) on slower, less‑pipelined units.
+ Implement “narrow” or specialized ops (e.g. 32‑bit mul, 52‑bit IFMA) on high‑performance units tuned for the workloads they care about (crypto, big‑int, etc.).
That’s why VPMULLQ and the synthesized sequence don’t share the same latency/throughput numbers even though they do equivalent math.

2. Microarchitectural design trade‑offs
CPUs have to balance:

+ Frequency: How fast the core clocks.
+ Area: How much silicon each unit consumes.
+ Power / thermals.
A very fast, fully pipelined 64‑bit integer vector multiplier per port would be:

Expensive in area.
Stressful for frequency and power.
Instead, Intel typically does something like:

Provide fast 32‑bit integer multiply (e.g. backing VPMULUDQ) on main integer/vector pipelines.
Provide fast 52‑bit multiply‑accumulate (VPMADD52*) on a dedicated crypto/big‑int oriented datapath.
Implement 64‑bit vector integer multiply (VPMULLQ) on a slower or less‑pipelined path, resulting in:
High latency (e.g. 15 cycles).
Mediocre throughput (e.g. one every 1.5 cycles).
From compiler/tuning perspective, that means:

The hardware design choice, driven by cost/perf trade‑offs, directly creates the latency gap you see in uops.info.
3. Decomposing a wide multiply into cheaper pieces
A 64‑bit multiply can be decomposed into smaller multiplies. For example, with 32‑bit halves:

Let 
```
a = a_hi<<32 + a_lo, b = b_hi<<32 + b_lo.
```
Then:
```
a * b = (a_lo * b_lo)
      + ((a_lo * b_hi + a_hi * b_lo) << 32)
      + (a_hi * b_hi << 64)
```
Each term is a 32×32→64 product (something VPMULUDQ is good at). You then:

Compute the partial products with VPMULUDQ or VPMADD52*.
Add/shift them appropriately to form the full 64‑bit result.
Why can this be faster?

Each partial multiply has shorter latency (3–4 cycles vs. 15).
Several partial multiplies may be independent and can run in parallel on different ports.
The follow‑up adds have low latency (1 cycle) and high throughput.
So even though you issue multiple instructions, the critical path (longest dependency chain) can be shorter than a single big, slow VPMULLQ.

4. Critical‑path vs. total work
Compilers, and the PR you linked, care a lot about critical‑path latency, not just instruction count:

A single VPMULLQ might look “cheaper” in terms of static instruction count.
But if it has a 15‑cycle dependency, it lengthens the loop’s recurrence or the dependence chain in your algorithm.
A 3‑ or 4‑instruction sequence, each with 3–4 cycle latency and some overlap, might reduce the effective chain to, say, ~6–8 cycles.
On a deeply pipelined, superscalar core:

VPMULLQ = one semantic op backed by a big, slow unit.
VPMADD52L/VPMULUDQ sequence = multiple semantic ops backed by smaller, faster, better‑pipelined units.




## Gantt Chart Assumptions

- Each row: instructions for one iteration (i, i+1, …).
- Time flows left → right in **cycles**.
- `=` is “instruction in flight”.
- `*` marks when the **result is ready** and can be used by the next iteration.
- Latencies:
  - `VPMULLQ` = 15 cycles
  - `Fast MUL` (`VPMULUDQ` / `VPMADD52*`) = 4 cycles
  - `ADD` (`VPADDQ` etc.) = 1 cycle
- We assume enough ports / frontend bandwidth that throughput is not the limiting factor; only the **dependency chain** matters.

## Case A – Single `VPMULLQ` with 15‑cycle latency

Per iteration:

```asm
; acc_i is available at cycle start
VPMULLQ acc_i, acc_i, mul   ; latency 15
; acc_(i+1) = result
```

### Timeline (iterations i, i+1, i+2)

```text
Cycles →
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 ...

Iter i:
  VPMULLQ  [===============*]                              
           0               15
           (acc_i -> acc_(i+1) ready at cycle 15)

Iter i+1:
  wait for acc_(i+1) ------> VPMULLQ  [===============*]    
                             15      16              30

Iter i+2:
  wait for acc_(i+2) -------------------------------> VPMULLQ  [===============*]
                                                      30      31              45
```

Interpretation:

- Iteration `i` starts at cycle 0, result ready at **cycle 15**.
- Iteration `i+1` can’t start its mul until cycle 15 (needs `acc_(i+1)`), and its result is ready at **cycle 30**.
- Iteration `i+2` starts at cycle 30, result at **cycle 45**.
- **Recurrence distance ≈ 15 cycles per iteration.**

The core cannot move the dependent chain faster than ~1 iteration per 15 cycles.

## Case B – Decomposed multiply (two fast muls + adds)

Now suppose each iteration does something like:

```asm
; acc_i is split conceptually into parts acc_lo, acc_hi
VPMULUDQ t0, acc_lo, mul_lo   ; fast MUL #1, latency 4
VPMULUDQ t1, acc_hi, mul_hi   ; fast MUL #2, latency 4, independent
VPADDQ   t2, t0, t1           ; ADD #1, latency 1
VPADDQ   acc_(i+1), t2, extra ; ADD #2, latency 1 (e.g. handling carries/other term)
; acc_(i+1) ready after this
```

### Single iteration chain

If `acc_i` is available at cycle `T`:

- `Fast MUL`s run from cycles `T–T+4`.
- First ADD runs at `T+4`, done at `T+5`.
- Second ADD runs at `T+5`, done at `T+6`.

So `acc_(i+1)` is ready at around **T+6**.

### Timeline (iterations i, i+1, i+2)

```text
Cycles →
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ...

Iter i:
  MULs:   [====]                              
          0    4
  ADD1:        [=]                            
                   4 5
  ADD2:          [=]*                         
                     5 6  (acc_(i+1) ready at cycle 6)

Iter i+1:
  MULs:           [====]                      
                  6    10
  ADD1:                [=]                    
                           10 11
  ADD2:                  [=]*                 
                             11 12 (acc_(i+2) ready at cycle 12)

Iter i+2:
  MULs:                     [====]           
                            12   16
  ADD1:                          [=]         
                                     16 17
  ADD2:                            [=]*     
                                       17 18 (acc_(i+3) ready at cycle 18)
```

More visually with alignment:

```text
Cycles →
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...

Iter i:
  MULs:   [====]                             
          0    4
  ADD1:        [=]                           
                   4 5
  ADD2:          [=]*                        
                     5 6  (acc_(i+1) ready at 6)

Iter i+1:
  wait -->         MULs:   [====]           
                    6     10
                  ADD1:        [=]          
                               10 11
                  ADD2:          [=]*       
                                   11 12 (acc_(i+2) ready at 12)

Iter i+2:
  wait -->                     MULs:   [====]
                                  12   16
                                ADD1:      [=]
                                          16 17
                                ADD2:        [=]*
                                               17 18 (acc_(i+3) ready at 18)
```

Interpretation:

- Iteration `i` starts at 0, result at **6**.
- Iteration `i+1` starts at 6, result at **12**.
- Iteration `i+2` starts at 12, result at **18**.
- **Recurrence distance ≈ 6 cycles per iteration.**

---

## Side‑by‑side comparison of recurrence distance

- **VPMULLQ version:**  
  - `acc_(i+1)` ready at `T + 15`  
  - Dependent iterations spaced ~15 cycles apart.

- **Decomposed version (2× fast mul + adds):**  
  - `acc_(i+1)` ready at `T + ~6`  
  - Dependent iterations spaced ~6 cycles apart.

Even though the decomposed version uses more instructions, the **longest dependency chain** per iteration is shorter, 
and some work (the two muls) is parallelizable. That’s the core reason why the IFMA/VPMULUDQ‑style sequence can 
beat a single `VPMULLQ` on “slow VPMULLQ” targets.
