# X86.td – Subtarget Features, Tunings, and Processor Feature Lists

- What a **`SubtargetFeature`** is in TableGen.
- The difference between **ISA features** and **tuning features**.
- How **micro-architectural “levels”** (x86-64-v1..v4) are expressed.
- How **CPU-specific feature lists** (Nehalem, Haswell, Skylake, Zen, etc.) are built by combining:
  - Base feature lists
  - Additional features
  - Tuning feature lists
  - Additional tunings / removed tunings

## 1. Subtarget features in TableGen
At the core of `X86.td` is the `SubtargetFeature` TableGen class (from `llvm/Target/Target.td`):
```tablegen
class SubtargetFeature<string Name,
                       string FieldName,
                       string Value,
                       string Desc,
                       list<SubtargetFeature> Implies = []>;
```

- **`Name`**: The `-mattr` / feature string used by frontends and users (e.g. `"sse4.2"`, `"avx2"`, `"fast-gather"`).
- **`FieldName`**: The C++ subtarget field or enum that this feature controls (e.g. `HasSSE4A`, `X86SSELevel`, `HasFastGather`).
- **`Value`**: The value to write into that field. For booleans this is typically `"true"` or `"false"`, for enums a named value (e.g. `"AVX2"`).
- **`Desc`**: Human-readable description (used for help output).
- **`Implies`**: Optional list of other features that this feature automatically enables when turned on.

### 1.1 Simple boolean feature example

```tablegen
def FeaturePOPCNT : SubtargetFeature<"popcnt", "HasPOPCNT", "true",
                                     "Support POPCNT instruction">;
```

- Enabling `+popcnt` causes `X86Subtarget::HasPOPCNT` to be set to `true`.
- Backends then know they may emit the `POPCNT` instruction.

On the command line:

- `llc -mtriple=x86_64 -mattr=+popcnt`
- Clang: `clang -target x86_64 -mattr=+popcnt ...`

### 1.2 Feature with implications example

```tablegen
def FeatureXSAVE   : SubtargetFeature<"xsave", "HasXSAVE", "true",
                                      "Support xsave instructions">;

def FeatureXSAVEOPT: SubtargetFeature<"xsaveopt", "HasXSAVEOPT", "true",
                                      "Support xsaveopt instructions",
                                      [FeatureXSAVE]>;
```

- Enabling `+xsaveopt` also implies `+xsave`.
- So if the user passes `-mattr=+xsaveopt`, both `HasXSAVEOPT` and `HasXSAVE` become `true`.


## 2. High-level categories in `X86.td`

The file is logically split into these main parts:

1. **Subtarget state** (16/32/64-bit modes)
2. **ISA features** (instruction-set capabilities, like SSE/AVX/AMX, SHA, etc.)
3. **Security / mitigation features** (retpoline, LVI, SESes, SLS hardening)
4. **Tuning features** (performance quirks or micro-optimizations)
5. **Processor Families** (very small set; largely historical)
6. **Register info and instruction descriptions**
7. **Scheduling models**
8. **`ProcessorFeatures` block**: predefined feature and tuning lists for:
   - x86-64-v[1–4] levels
   - Intel µarches (Nehalem, Haswell, Skylake, Icelake, Alder Lake, Sapphire Rapids, Granite Rapids, Arrow Lake, etc.)
   - AMD µarches (Barcelona, Bobcat, Bulldozer, Zen1–Zen6)
   - Atom families (Silvermont, Goldmont, …)

focuses on parts (2), (4), and (8).

## 3. Subtarget state: Is16Bit / Is32Bit / Is64Bit / IsX32
At the top you have “mode” features:
```tablegen
def Is64Bit : SubtargetFeature<"64bit-mode", "Is64Bit", "true",
                               "64-bit mode (x86_64)">;
def Is32Bit : SubtargetFeature<"32bit-mode", "Is32Bit", "true",
                               "32-bit mode (80386)">;
def Is16Bit : SubtargetFeature<"16bit-mode", "Is16Bit", "true",
                               "16-bit mode (i8086)">;
def IsX32   : SubtargetFeature<"x32", "IsX32", "true",
                               "64-bit with ILP32 programming model (e.g. x32 ABI)">;
```

These do *not* describe the ISA extensions, but the **execution mode / ABI**.

Example:

- For `x86_64-unknown-linux-gnu`, the subtarget for codegen would set `Is64Bit=true`.
- For `i386-unknown-linux-gnu`, codegen would set `Is32Bit=true`.
- For the x32 ABI, both `Is64Bit` and `IsX32` may be set.

## 4. ISA features (instruction-set capabilities)

These are the bulk of the early part of the file (`FeatureX87`, `FeatureSSE*`, `FeatureAVX*`, `FeatureAMX*`, security instructions, etc.).

### 4.1 Example: SSE and AVX levels

```tablegen
def FeatureSSE1  : SubtargetFeature<"sse",   "X86SSELevel", "SSE1",
                                    "Enable SSE instructions">;

def FeatureSSE2  : SubtargetFeature<"sse2",  "X86SSELevel", "SSE2",
                                    "Enable SSE2 instructions", [FeatureSSE1]>;

def FeatureSSE3  : SubtargetFeature<"sse3",  "X86SSELevel", "SSE3",
                                    "Enable SSE3 instructions", [FeatureSSE2]>;

def FeatureSSSE3 : SubtargetFeature<"ssse3", "X86SSELevel", "SSSE3",
                                    "Enable SSSE3 instructions", [FeatureSSE3]>;

def FeatureSSE41 : SubtargetFeature<"sse4.1", "X86SSELevel", "SSE41",
                                    "Enable SSE 4.1 instructions", [FeatureSSSE3]>;

def FeatureSSE42 : SubtargetFeature<"sse4.2", "X86SSELevel", "SSE42",
                                    "Enable SSE 4.2 instructions", [FeatureSSE41]>;
```

- All these features write an enum value into `X86Subtarget::X86SSELevel`.
- Each higher level *implies* the previous one (e.g. SSE4.2 ⇒ SSE4.1 ⇒ SSSE3 ⇒ SSE3 ⇒ SSE2 ⇒ SSE1).
- If you pass `-mattr=+sse4.2`, the backend can rely on all SSE1–SSE4.2 instructions being available.

Similarly for AVX:

```tablegen
def FeatureAVX   : SubtargetFeature<"avx",  "X86SSELevel", "AVX",
                                    "Enable AVX instructions",
                                    [FeatureSSE42]>;

def FeatureAVX2  : SubtargetFeature<"avx2", "X86SSELevel", "AVX2",
                                    "Enable AVX2 instructions",
                                    [FeatureAVX]>;

def FeatureAVX512: SubtargetFeature<"avx512f", "X86SSELevel", "AVX512",
                                    "Enable AVX-512 instructions",
                                    [FeatureAVX2, FeatureFMA, FeatureF16C]>;
```

### 4.2 Example: Auxiliary scalar / system instructions

Many features map to individual instructions or small groups:

```tablegen
def FeatureMOVBE  : SubtargetFeature<"movbe", "HasMOVBE", "true",
                                     "Support MOVBE instruction">;

def FeatureLZCNT  : SubtargetFeature<"lzcnt", "HasLZCNT", "true",
                                     "Support LZCNT instruction">;

def FeatureSHA    : SubtargetFeature<"sha", "HasSHA", "true",
                                     "Enable SHA instructions",
                                     [FeatureSSE2]>;
```

These typically get set either directly from the CPU model (e.g. `-mcpu=haswell`) or via `-mattr`.

## 5. Security / mitigation features

These features do not add ISA instructions per se but change **code generation strategy** to mitigate side-channel vulnerabilities.

Examples:

```tablegen
def FeatureRetpolineIndirectCalls
  : SubtargetFeature<"retpoline-indirect-calls",
                     "UseRetpolineIndirectCalls", "true",
                     "Remove speculation of indirect calls from the generated code">;

def FeatureRetpolineIndirectBranches
  : SubtargetFeature<"retpoline-indirect-branches",
                     "UseRetpolineIndirectBranches", "true",
                     "Remove speculation of indirect branches from the generated code">;

def FeatureRetpoline
  : SubtargetFeature<"retpoline", "DeprecatedUseRetpoline", "true",
                     "...",
                     [FeatureRetpolineIndirectCalls,
                      FeatureRetpolineIndirectBranches]>;
```

- Enabling `+retpoline` is a **deprecated umbrella** that turns on both `UseRetpolineIndirectCalls` and `UseRetpolineIndirectBranches`.

Similarly for LVI and SESes:

```tablegen
def FeatureLVIControlFlowIntegrity
  : SubtargetFeature<"lvi-cfi", "UseLVIControlFlowIntegrity", "true",
                     "...">;

def FeatureSpeculativeExecutionSideEffectSuppression
  : SubtargetFeature<"seses", "UseSpeculativeExecutionSideEffectSuppression",
                     "true", "...",
                     [FeatureLVIControlFlowIntegrity]>;
```

## 6. Tuning features (performance characteristics)

Under the comment:

```tablegen
//===----------------------------------------------------------------------===//
// X86 Subtarget Tuning features
//===----------------------------------------------------------------------===//
```

The file lists many features prefixed with `Tuning...`. These describe performance quirks and preferences, not ISA availability.

Typical pattern:

```tablegen
def TuningSlowDivide64 : SubtargetFeature<"idivq-to-divl",
                                          "HasSlowDivide64", "true",
                                          "Use 32-bit divide for positive values less than 2^32">;
```

- This does not say whether `idivq` exists (it does for 64-bit); it says **it is slow on this CPU**.
- The instruction selector / DAG combiner can then prefer cheaper sequences (e.g. 32-bit divides and extends) when this tuning is on.

Some representative examples:

### 6.1 Latency- or throughput-related tunings

```tablegen
def TuningSlowSHLD : SubtargetFeature<"slow-shld", "IsSHLDSlow", "true",
                                      "SHLD instruction is slow">;

def TuningFastScalarFSQRT
  : SubtargetFeature<"fast-scalar-fsqrt", "HasFastScalarFSQRT", "true",
                     "Scalar SQRT is fast (disable Newton-Raphson)">;

def TuningFastVectorFSQRT
  : SubtargetFeature<"fast-vector-fsqrt", "HasFastVectorFSQRT", "true",
                     "Vector SQRT is fast (disable Newton-Raphson)">;
```

Codegen effects:

- If `HasFastScalarFSQRT` is true, the backend will prefer a single `SQRTSS` instead of `RSQRT + Newton-Raphson iteration`.
- `IsSHLDSlow` might cause the backend to prefer rotate instructions or shifts+or sequences instead of `SHLD`.

### 6.2 Micro-architectural quirks

```tablegen
def TuningSlowTwoMemOps : SubtargetFeature<"slow-two-mem-ops",
                                           "SlowTwoMemOps", "true",
                                           "Two memory operand instructions are slow">;
```

- On such CPUs, codegen tries to avoid `CALL [mem]`, `PUSH [mem]` etc.; instead it emits `MOV [mem] -> reg` followed by `CALL reg`, which can be faster.

```tablegen
def TuningSlow3OpsLEA : SubtargetFeature<"slow-3ops-lea", "Slow3OpsLEA", "true",
                                         "LEA instruction with 3 ops or certain registers is slow">;
```

- Codegen avoids LEA with all of base/index/offset, or LEA using EBP/RBP/R13 as base, and may try to split into separate adds/shifts.

### 6.3 False dependencies and domain-crossing issues

```tablegen
def TuningPOPCNTFalseDeps : SubtargetFeature<"false-deps-popcnt",
                                             "HasPOPCNTFalseDeps", "true",
                                             "POPCNT has a false dependency on dest register">;

def TuningLZCNTFalseDeps : SubtargetFeature<"false-deps-lzcnt-tzcnt",
                                            "HasLZCNTFalseDeps", "true",
                                            "LZCNT/TZCNT have a false dependency on dest register">;
```

- With these tunings, when the backend emits `POPCNT` or `LZCNT` instructions, it can insert `xor reg, reg` or use a new virtual register to break false dependencies.

```tablegen
def TuningFastGather : SubtargetFeature<"fast-gather", "HasFastGather", "true",
                                        "Indicates if gather is reasonably fast ...">;
```

- With `HasFastGather=true`, vectorizer / cost model can consider lowering scatter/gather patterns into actual `VGATHER` instructions; otherwise, it may prefer scalarizes or explicit loads.


## 7. Additional tunings: enabling / removing per CPU

Later, inside the `ProcessorFeatures` block, tunings are combined into per-CPU lists. Some CPUs **add** tunings and also **remove** some inherited tunings.

Example: Alder Lake tuning:

```tablegen
list<SubtargetFeature> ADLAdditionalTuning = [
  TuningPERMFalseDeps,
  TuningPreferMovmskOverVTest,
  TuningFastImmVectorShift,
  TuningSlowPMULLQ
];

list<SubtargetFeature> ADLRemoveTuning = [TuningPOPCNTFalseDeps];

list<SubtargetFeature> ADLTuning =
    !listremove(!listconcat(SKLTuning, ADLAdditionalTuning),
                ADLRemoveTuning);
```

- `ADLTuning` starts from `SKLTuning`.
- Then adds:
  - `TuningPERMFalseDeps`
  - `TuningPreferMovmskOverVTest`
  - `TuningFastImmVectorShift`
  - `TuningSlowPMULLQ`
- Then removes `TuningPOPCNTFalseDeps` (POPCNT false-dependency issue no longer applies).

In TableGen `!listconcat` and `!listremove` operate on lists:

- `!listconcat(A, B)` → concatenation of lists.
- `!listremove(L, ToRemove)` → `L` minus elements present in `ToRemove`.


## 8. X86-64 micro-architecture levels: x86-64-v1 to v4

Inside `def ProcessorFeatures { ... }` you’ll see the standardized **x86-64 levels**:

```tablegen
list<SubtargetFeature> X86_64V1Features = [
  FeatureX87, FeatureCX8, FeatureCMOV, FeatureMMX, FeatureSSE2,
  FeatureFXSR, FeatureNOPL, FeatureX86_64,
];

list<SubtargetFeature> X86_64V1Tuning = [
  TuningMacroFusion,
  TuningSlow3OpsLEA,
  TuningSlowDivide64,
  TuningSlowIncDec,
  TuningInsertVZEROUPPER
];
```

Level 2 builds on level 1:

```tablegen
list<SubtargetFeature> X86_64V2Features =
  !listconcat(X86_64V1Features, [
    FeatureCX16, FeatureLAHFSAHF64, FeatureCRC32, FeaturePOPCNT,
    FeatureSSE42
  ]);

list<SubtargetFeature> X86_64V2Tuning = [
  TuningMacroFusion,
  TuningSlow3OpsLEA,
  TuningSlowDivide64,
  TuningSlowUAMem32,
  TuningFastScalarFSQRT,
  TuningFastSHLDRotate,
  TuningFast15ByteNOP,
  TuningPOPCNTFalseDeps,
  TuningInsertVZEROUPPER
];
```

Level 3:

```tablegen
list<SubtargetFeature> X86_64V3Features =
  !listconcat(X86_64V2Features, [
    FeatureAVX2, FeatureBMI, FeatureBMI2, FeatureF16C, FeatureFMA,
    FeatureLZCNT, FeatureMOVBE, FeatureXSAVE
  ]);

list<SubtargetFeature> X86_64V3Tuning = [
  TuningMacroFusion,
  TuningSlow3OpsLEA,
  TuningSlowDivide64,
  TuningFastScalarFSQRT,
  TuningFastSHLDRotate,
  TuningFast15ByteNOP,
  TuningFastVariableCrossLaneShuffle,
  TuningFastVariablePerLaneShuffle,
  TuningPOPCNTFalseDeps,
  TuningLZCNTFalseDeps,
  TuningInsertVZEROUPPER,
  TuningAllowLight256Bit
];
```

Level 4:

```tablegen
list<SubtargetFeature> X86_64V4Features =
  !listconcat(X86_64V3Features, [
    FeatureBWI,
    FeatureCDI,
    FeatureDQI,
    FeatureVLX,
  ]);

list<SubtargetFeature> X86_64V4Tuning = [
  TuningMacroFusion,
  TuningSlow3OpsLEA,
  TuningSlowDivide64,
  TuningFastScalarFSQRT,
  TuningFastVectorFSQRT,
  TuningFastSHLDRotate,
  TuningFast15ByteNOP,
  TuningFastVariableCrossLaneShuffle,
  TuningFastVariablePerLaneShuffle,
  TuningPrefer256Bit,
  TuningFastGather,
  TuningPOPCNTFalseDeps,
  TuningInsertVZEROUPPER,
  TuningAllowLight256Bit
];
```

These lists are used when you target:

- `-march=x86-64`      → V1 baseline
- `-march=x86-64-v2`   → V2 features, V2 tuning
- `-march=x86-64-v3`   → V3 features, V3 tuning
- `-march=x86-64-v4`   → V4 features, V4 tuning

### Example: computing X86_64V3Features

Conceptually:

```text
X86_64V1Features =
  { X87, CX8, CMOV, MMX, SSE2, FXSR, NOPL, X86_64 }

X86_64V2Features =
  X86_64V1Features ∪ { CX16, LAHFSAHF64, CRC32, POPCNT, SSE4.2 }

X86_64V3Features =
  X86_64V2Features ∪
  { AVX2, BMI, BMI2, F16C, FMA, LZCNT, MOVBE, XSAVE }
```

## 11. “Additional features” and “additional tuning” patterns

Throughout `ProcessorFeatures` you see a repeating pattern:

```tablegen
list<SubtargetFeature> XXXAdditionalFeatures = [...];
list<SubtargetFeature> XXXTuning = [... or based on previous tuning ...];
list<SubtargetFeature> XXXFeatures =
  !listconcat(PreviousFeatures, XXXAdditionalFeatures);
```

Sometimes:

```tablegen
list<SubtargetFeature> XXXAdditionalTuning = [...];
list<SubtargetFeature> XXXRemoveTuning = [...];
list<SubtargetFeature> XXXTuning =
  !listremove(!listconcat(PreviousTuning, XXXAdditionalTuning),
              XXXRemoveTuning);
```

This gives a **compositional** way to define each CPU:

- Start from a *base* CPU (e.g. Nehalem, Zen, Skylake).
- Add new ISA features introduced in this generation.
- Add tunings that describe new performance behavior.
- Optionally drop (remove) tunings that are no longer accurate.

## 12. How all this is used in practice

### 12.1 From `-mcpu` / `-march` to features/tuning

The driver (Clang/llc) maps `-mcpu=haswell` to:

1. `HSWFeatures` as the **feature set**.
2. `HSWTuning` as the **tuning set**.

When building `X86Subtarget`, these features:

- Set boolean and enum fields like `HasAVX2`, `X86SSELevel`, `HasFastGather`, `Slow3OpsLEA`, etc.
- Are used by:
  - Instruction selector / DAG combiner
  - Machine scheduler
  - Cost-model driven transforms (in vectorizer, loop unrolling, inlining, etc.)

