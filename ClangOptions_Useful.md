# Search Clang Doc
+ [https://clang.llvm.org/docs/search.html](Clang Option Search)
+ https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
+ https://gcc.gnu.org/onlinedocs/gcc/Invoking-GCC.html
+ https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
+ How to repro bug FE/MIDDLE END /BE with LTO and Windows https://llvm.org/docs/HowToSubmitABug.html
# LLVM ENABLE CRASH DIAGNOSTIC
+ `export LLVM_ENABLE_DUMP=1`
# Print All passes enabled
```
$opt=-O0
echo 'int;' | clang -xc $opt  - -o /dev/null -###
```
+ https://clang.llvm.org/docs/ClangCommandLineReference.html
# Print clang options, default values
```
echo 'int;' | clang -xc - -o /dev/null -mllvm -print-all-options
clang -help
clang -cc1 -help
clang -cc1 -mllvm -help
clang -cc1 -mllvm -help-list-hidden
clang -cc1as -help
```
## preprocessor
+ Dump all macros set by different options
```
clang -E -dM  <source>   # does not need emit-llvm etc.
-E: stop before compiling, after preprocessing, produces .i
-S: stop before assembling, produces .s
-c: stop before linking, produces .o
-v: print commands executed and run
-###: print commands executed, don't run
-o -: print to stdout rather than write output to file
-fno-discard-value-names: identifiers from source rather than numbers.
-g0: less debug info in IR.
```
# IR 
```
clang -O0 -emit-llvm -Xclang -disable-llvm-passes -S  bug.c -o bug.ll
opt bug.ll -o bug.opt.ll -passes='sroa,instcombine,loop(loop-rotate,indvars),dot-cfg' -S
```

# CLANG/LLC
 + To pass target specific option to clang and llc
```
clang -mavx/-mavx512f test.c -S
llc -mattr=avx512f/-mattr=+avx512f test.ll -S
```
+ MCPU / MARCH / MTUNE
  
  ```
  https://maskray.me/blog/2022-08-28-march-mcpu-mtune
  -march=X: (execution domain) Generate code that can use instructions available in the architecture X
  -mtune=X: (optimization domain) Optimize for the microarchitecture X, but does not change the ABI or make assumptions about available instructions
  -mcpu=X: Specify both -march= and -mtune= but can be overridden by the two options. The supported values are generally the same as -mtune=. The architecture name is inferred from X
   simplest case where only one option is desired, use -march= for x86 and -mcpu= for other targets.
  When optimizing for the local machine, just use -march=native for x86 and -mcpu=native for other targets.
  When the architecture and microarchitecture are both specified, i.e. when both the execution domain and the optimization domain need to be specified, specify -march= and -mtune=, and avoid -mcpu=.
  https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/compiler-flags-across-architectures-march-mtune-and-mcpu
  ```
+ Compiler Driver and cross compilation https://maskray.me/blog/2021-03-28-compiler-driver-and-cross-compilation#clang
# Assembly
+ https://stackoverflow.com/questions/38552116/how-to-remove-noise-from-gcc-clang-assembly-output/38552509#38552509 
`g++ -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -fverbose-asm  -Wall -Wextra  foo.cpp   -O3 -masm=intel -S -o- | less`
+ https://panthema.net/2013/0124-GCC-Output-Assembler-Code/
` gcc test.c -o test -Wa,-adhln=test-O3.s -g -fverbose-asm -masm=intel -O3 -march=native`

# DUMP AST
```
clang -Xclang -ast-dump -ast-dump -ast-dump-filter=func
clang -cc1 -ast-dump -ast-dump -ast-dump-filter=func
```
# Export DOT  for CFG/Callgraph
```
opt test2.ll -passes=dot-callgraph -callgraph-dot-filename-prefix=callgraph -disable-output
opt -passes=dot-cfg test.ll -cfg-dot-filename-prefix=temp -o /dev/null
dot -Tpng temp.foo.dot -o file.png && sxiv file.png
dot -Tpdf cfg.dot -o cfg.pdf
```
# Graphviz
- https://www.graphviz.org/doc/info/command.html

#  Generate ASM from Compiler

# Remove cfi and noise
`clang -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti `

# OPT
+ How Pass manager executes and dependent passes (New PM), old PM uses -debug-pass=Structure
`opt -debug-pass-manager`

## Run IR CFG DOT
`opt -passes="view-cfg" -cfg-func-name=<func> <ll filename> -disable-output`
+ LLVM IR naming 
- https://clang.llvm.org/docs/UsersManual.html#controlling-value-names-in-llvm-ir
- instnamer https://llvm.org/docs/Passes.html#id93
- Randomizer for name https://llvm.org/doxygen/MetaRenamer_8cpp_source.html

# Xlinker Options
# IR debug and print-after-all from clang
`clang -mllvm --print-before-all   -mllvm --filter-print-funcs=KERNEL_FUNC`
# IR-LTO on, dump debug and print-after-all
`clang -Xlinker -plugin-opt=-print-before-all   -Xlinker -plugin-opt=-filter-print-funcs=KERNEL_FUNC`
# Same as above but list
`clang -Wl,-mllvm,-print-after-all -Wl,-mllvm,-filter-print-funcs=KERNEL_FUNC`

# Opt Reports
```
clang -O3 -foptimization-record-file=kernel.yaml; opt-viewer.py kernel.yaml
clang -O3 -ftime-trace
opt -O3 –passes="my-opt" –pass-remarks-output=remark.yaml -disable-output
```

# Print IR Debug Options
+ [https://llvm.org/doxygen/PrintPasses_8cpp_source.html](All options)
+ [https://llvm.org/doxygen/StandardInstrumentations_8cpp_source.html](IR debug options)
+ print-changed=[dot-cfg | dot-cfg-quiet]
+ print-before-pass-number
+ print-on-crash-path
+ print-on-crash
```
clang test.c -O2 -mllvm -print-changed
opt test.ll -O2 -print-changed
llc test.ll  -print-changed
```

## PERF ISSUE OPTIONS
# FE Alignment Options
```
-malign-branch-boundary=16
-mbranches-within-32B-boundaries
-mpad-max-prefix-size=5
-malign-branch=fused,jcc,jmp
-mllvm -align-all-functions=5
-mllvm -align-all-blocks=5
```
# Machine Block Placement 
+ https://github.com/llvm-mirror/llvm/blob/master/lib/CodeGen/MachineBlockPlacement.cpp

# LTO Debugging LD + plugin GOLD/BFD (TODO)
+  Dump LTO IRs 
  `clang -flto -Wl,-plugin-opt=save-temps`
+ Make the crash produce a reproducible artifact
  `clang++ -flto -fuse-ld=lld -Wl,--reproduce=./lto-repro.tar`
+ Thin LTO
  `-Wl,--plugin-opt=thinlto-save-temps`
# Windows Specific
# lld-linker Diagnostics
`export FORCE_LLD_DIAGNOSTICS_CRASH=1`

# CRASH LOG SYMBOLIZE  (TODO)
 `export LLVM_SYMBOLIZER_PATH=/path/to/llvm-symbolizer`

## LLVM-MCA
- https://llvm.org/docs/CommandGuide/llvm-mca.html
- `clang foo.c -O2 --target=x86_64 -S -o - | llvm-mca -mcpu=btver2 --timeline`
- `llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=3 -timeline dot-product.s` 
- https://godbolt.org/z/3b6G9x98j
  
# Using Tablegen
+ <GENERATOR> like `gen-global-isel`  etc describe in `llvm/lib/Target/X86/CMakeLists.txt`
+ `llvm-tblgen -gen-global-isel  -I$LLVM_TOP/llvm/lib/Target/X86 -I$LLVM_TOP/llvm/include -I$LLVM_TOP/llvm/lib/Target $LLVM_TOP/llvm/lib/Target/X86/X86.td --write-if-changed -o <OUTPUT>`
+ add '--debug-only=gisel-emitter` and `--debug` 
+ Supports statistics

## GCC Options
* Check Default options 
  + Check default target options/features enabled or disabled  `-Q --help=target`
  + Check enabled optimization configurations `-Q --help=optimizers`
  + Tool flow how GCC invokes all its subprocesses (preprocessor, compiler)  `-dumpspecs `
  + https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html
* Debugging Options for GCC (Opt reports, dumps  after stage)
 + https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html
 
