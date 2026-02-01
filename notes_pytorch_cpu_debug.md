### To debug PyTorch’s **CPU backend**  (JIT/inductor-generated code compiled by a C++ compiler at runtime)
+ force the kernel to be emitted to disk
+ capture the exact compile command
+ rebuild with debug symbols/no optimizations
+ attach a debugger either to the Python process or to the compiler/spawned process.

## Steps
### “compiled on kernel”  in PyTorch CPU 
- **TorchInductor (PyTorch 2.x `torch.compile`) CPU**: generates C++ (or LLVM) and compiles it with `g++/clang` at runtime.
- **cpp_extension / custom ops**: builds C++ at runtime and loads a `.so`.
- **oneDNN/MKLDNN/JITed primitives**: less common “generated” code paths (but still debuggable with env logging + symbols).

### Debugging **Inductor CPU**
+ Build PyTorch in a debug  (so backtraces make sense for pytorch but not needed for kernels)
+ Make Inductor write generated sources and keep build artifacts
For TorchInductor, before running your repro Environment variables (most useful):
```
- `TORCH_LOGS=inductor` (or more verbose: `TORCH_LOGS="+inductor,+output_code"` depending on version)
- `TORCHINDUCTOR_VERBOSE=1`
- `TORCHINDUCTOR_DEBUG=1` (commonly causes extra dumping)
- `TORCH_COMPILE_DEBUG=1` (often enables extra dumps)
- `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache` (or any directory you control)
```
+ https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html
+ `torch._dynamo.config.verbose = True` in python script also works

+ Reproduce with  freezed variability
  - set `torch.manual_seed(0)`
  - fix input shapes/dtypes
  - run a warmup then one iteration that triggers compilation

Example skeleton:

```python
import torch
torch.manual_seed(0)

@torch.compile
def f(x):
    return (x.sin() * x.cos()).sum()

x = torch.randn(1024, 1024)
print(f(x))
```

Run once to compile, then run again to execute from cache.

+ Attach debugger to the *Python process* to debug execution of the compiled kernel
Once the kernel is compiled and loaded, it’s just native code running inside the Python process.

+ How to set breakpoints in generated code
You usually need **debug symbols** in the generated `.so`. By default, Inductor might compile with optimizations and without `-g`.
then we need extra step, rebuild with following and load again.
  - ```bash
    export CFLAGS="-O0 -g -fno-omit-frame-pointer"
    export CXXFLAGS="-O0 -g -fno-omit-frame-pointer"
    ```
  -  clear cache or change cache dir
    
+ see Inductor C++ Builder for more

+  Debug *the compilation step itself* (compiler invoked at runtime)
- If compilation fails (missing header, wrong flags, ICE in compiler), debug the compiler invocation:
  - Make sure logs show the full compile command.
  - Re-run that exact command manually in the build dir.
  - To trace subprocess execution:
   - `strace -f -o /tmp/trace.txt python repro.py` (Linux)
  - If Ninja is used, use `ninja -v`.
+ correctness issue-  compare eager vs compiled and bisect
For correctness bugs in CPU generated kernels:
- Run eager (no compile) and compiled, compare outputs.
- Narrow to a single op / smallest subgraph.
- If it’s Inductor, try toggles to isolate:
  - disable certain fusions
  - change scheduling/tiling heuristics
  - run with fewer threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

+ performance issue - use perf + inductor dumps
- Use `perf record -g -- python repro.py` then `perf report`
- Check thread settings:
  - `torch.set_num_threads(n)`
  - `OMP_NUM_THREADS`, `KMP_AFFINITY`

# CHECKLIST
+ using **`torch.compile`** (Inductor) on CPU  **C++ extension** (`torch.utils.cpp_extension`)
+ symptom: **crash**, **wrong output**, **compile error**, or **slow**?  
+ OS + compiler: Linux? clang or gcc?

# DEBUG
```
import os
import torch

# Make debugging easier / more deterministic on CPU
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

# (Optional) torch.compile debug toggles:
# 1) TorchDynamo verbose logging
# os.environ["TORCH_LOGS"] = "dynamo,inductor"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# 2) Dump Inductor-generated code (useful when backend="inductor")
# os.environ["TORCHINDUCTOR_DEBUG"] = "1"

device = "cpu"

# A tiny "kernel-like" function: elementwise + reduction + pointwise
# (kept simple but exercises codegen paths)
def simple_kernel(x: torch.Tensor, w: torch.Tensor):
    # x: [N, D], w: [D]
    y = x * w               # broadcast mul
    y = torch.relu(y)       # pointwise
    s = y.sum(dim=1)        # reduction
    out = s * 0.1 + 1.0     # pointwise
    return out              # [N]

# Compile for CPU.
# - backend="eager" is great to debug graph breaks while still going through Dynamo.
# - backend="inductor" will generate an optimized CPU kernel (and lets you inspect generated code with TORCHINDUCTOR_DEBUG=1).
compiled_eager = torch.compile(simple_kernel, backend="eager", fullgraph=False)
compiled_inductor = torch.compile(simple_kernel, backend="inductor", fullgraph=False)

# Inputs
N, D = 8, 16
x = torch.randn(N, D, device=device)
w = torch.randn(D, device=device)

# Run eager (baseline), compiled-eager, compiled-inductor
ref = simple_kernel(x, w)
out1 = compiled_eager(x, w)
out2 = compiled_inductor(x, w)

print("max|ref - compiled_eager|   =", (ref - out1).abs().max().item())
print("max|ref - compiled_inductor|=", (ref - out2).abs().max().item())

# If you want to force a graph break to see debugging behavior:
def kernel_with_graph_break(x):
    y = x + 1
    # Python-side data-dependent branch forces a break
    if y[0].item() > 0:
        y = y * 2
    return y

dbg = torch.compile(kernel_with_graph_break, backend="eager")
print(dbg(torch.randn(4)))
```
Simple Run
```
python3 kernel.py
max|ref - compiled_eager|   = 0.0
max|ref - compiled_inductor|= 1.1920928955078125e-07
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0] Graph break from `Tensor.item()`, consider setting:
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]     torch._dynamo.config.capture_scalar_outputs = True
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0] or:
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0] to include these operations in the captured graph.
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0] Graph break: from user code at:
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]   File "/mnt/c/work/pytorch_debug/kernel.py", line 50, in kernel_with_graph_break
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]     if y[0].item() > 0:
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]
W0201 11:45:18.591000 210770 torch/_dynamo/variables/tensor.py:869] [1/0]
tensor([-0.3360,  1.8871,  1.7680,  1.0571])
```
Debug Run with Exported Above variables
```
torch/_inductor/config.py:628] compile_threads set to 16
torch/_inductor/async_compile.py:147] Creating subprocess pool with 16 workers
torch/_inductor/config.py:628] compile_threads set to 16
torch/_inductor/codecache.py:905] [0/1] FX graph cache hash details for key ff5tzdurkhhdldzxihywjaapklrfql4ssnonbev5cdxzg3qfycqf:
torch/_inductor/codecache.py:905] [0/1] [z7sl3oqxcklyburarz63uaeg3dlljulis7z5mvuycoghwgmqixh] gm: <lambda>()
torch/_inductor/codecache.py:905] [0/1]
torch/_inductor/codecache.py:905] [0/1]
torch/_inductor/codecache.py:905] [0/1]
torch/_inductor/codecache.py:905] [0/1] def forward(self, arg0_1, arg1_1):
torch/_inductor/codecache.py:905] [0/1]     mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
torch/_inductor/codecache.py:905] [0/1]     relu = torch.ops.aten.relu.default(mul);  mul = None
torch/_inductor/codecache.py:905] [0/1]     sum_1 = torch.ops.aten.sum.dim_IntList(relu, [1]);  relu = None
torch/_inductor/codecache.py:905] [0/1]     mul_1 = torch.ops.aten.mul.Tensor(sum_1, 0.1);  sum_1 = None
torch/_inductor/codecache.py:905] [0/1]     add = torch.ops.aten.add.Tensor(mul_1, 1.0);  mul_1 = None
torch/_inductor/codecache.py:905] [0/1]     return (add,)
torch/_inductor/codecache.py:905] [0/1]
torch/_inductor/codecache.py:905] [0/1] # To see more debug info, please use `graph_module.print_readable()`
torch/_inductor/codecache.py:905] [0/1] [cx7y7j5kyzvruvnua4evc7nss2pdvezbfdc2bdjt76hwxocsojn] example_inputs[0]: TensorMetadata(dtype=torch.float32, shape=torch.Size([8, 16]), stride=(16, 1), device=device(type='cpu'), layout=torch.strided, memory_format=torch.contiguous_format, storage_offset=0, storage_bytes=None, requires_grad=False, is_quantized=False, is_conj=False, is_neg=False, is_inference=False, is_sparse=False, is_coalesced=None, dense_dim=None, sparse_dim=None)
torch/_inductor/codecache.py:905] [0/1] [2eca7tuphg6byxscx4yw3cib2we5s7qgi7lhcelnbipkpzqcode] example_inputs[1]: TensorMetadata(dtype=torch.float32, shape=torch.Size([16]), stride=(1,), device=device(type='cpu'), layout=torch.strided, memory_format=torch.contiguous_format, storage_offset=0, storage_bytes=None, requires_grad=False, is_quantized=False, is_conj=False, is_neg=False, is_inference=False, is_sparse=False, is_coalesced=None, dense_dim=None, sparse_dim=None)
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] cache_key_tag:
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] fx_kwargs[aot_mode]: False
torch/_inductor/codecache.py:905] [0/1] [lmglpn4zi7vob56n34r2j2rk7flv5xfgrcvmo7xcpirqsitygqx] fx_kwargs[boxed_forward_device_index]: BoxedDeviceIndex(value=None)
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] fx_kwargs[cpp_wrapper]: False
torch/_inductor/codecache.py:905] [0/1] [xq2hdkbfkbcuye6rgtypayrkhqf4cntij2dsd24rei3lsknakkf] fx_kwargs[cudagraphs]: BoxedBool(value=False)
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] fx_kwargs[extern_node_serializer]: None
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] fx_kwargs[is_backward]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] fx_kwargs[is_inference]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] fx_kwargs[layout_opt]: None
torch/_inductor/codecache.py:905] [0/1] [h25wqx6vliw4j5rtzzbv6latydxyei3deyg6v7wzvnzryfktuki] fx_kwargs[static_input_idxs]: []
torch/_inductor/codecache.py:905] [0/1] [du4vyrfyozrfxcf6kk6ma7oqwatapifazeelfsawmsiu6gjdtxp] deterministic_algorithms_settings: (False, False, True)
torch/_inductor/codecache.py:905] [0/1] [7as26aeta7rzhgm2mxh4el36kupf55fr27327kzc2fsdiy3nexy] cuda_matmul_settings: (True, True, True)
torch/_inductor/codecache.py:905] [0/1] [vyt5svtd5l64ntskurjjswwwzcsb54qs64fzb7yprgcrnad3rnl] torch_version: <bytes>
torch/_inductor/codecache.py:905] [0/1] [ou2myrupfm7jqj7zhckhwr35g4s7cslkzawm6jrbi4dt72allir] system_info[hash]: 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[TYPE_CHECKING]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[enable_auto_functionalized_v2]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[debug]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[disable_progress]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[verbose_progress]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[fx_graph_cache]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[fx_graph_remote_cache]: None
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[bundle_triton_into_fx_graph_cache]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[autotune_local_cache]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[autotune_remote_cache]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[bundled_autotune_remote_cache]: None
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_disable_caches]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[sleep_sec_TESTING_ONLY]: None
torch/_inductor/codecache.py:905] [0/1] [pikr7bbcoixfzftsazp5ggufhdklj24babfry77bl4nuvyrrcp4] inductor_config[custom_op_default_layout_constraint]: needs_fixed_stride_order
torch/_inductor/codecache.py:905] [0/1] [pikr7bbcoixfzftsazp5ggufhdklj24babfry77bl4nuvyrrcp4] inductor_config[triton_kernel_default_layout_constraint]: needs_fixed_stride_order
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp_wrapper]: False
torch/_inductor/codecache.py:905] [0/1] [b4ha3ravs3qv237q65hpfqegbnoww7tf2ahcbu2i7xo6te5spqs] inductor_config[c_shim_version]: 2
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[dce]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[static_weight_shapes]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[size_asserts]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[nan_asserts]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[pick_loop_orders]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[inplace_buffers]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[allow_buffer_reuse]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[memory_planning]: False
torch/_inductor/codecache.py:905] [0/1] [x75won4jmsgeb63pcvwr2y4eteyzzdhmf5rv6xhjppie4hx2yu5] inductor_config[memory_pool]: intermediates
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[benchmark_harness]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[epilogue_fusion]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[epilogue_fusion_first]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[pattern_matcher]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[b2b_gemm_pass]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[joint_custom_pre_pass]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[joint_custom_post_pass]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[pre_grad_custom_pass]: None
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[split_cat_fx_passes]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[efficient_conv_bn_eval_fx_passes]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[is_predispatch]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[group_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[batch_fusion]: True
torch/_inductor/codecache.py:905] [0/1] [4bryyl4ahh5whyg3zwqebpwmjnx6w77nqgqbdjlowju6lkqtn7w] inductor_config[pre_grad_fusion_options]: {}
torch/_inductor/codecache.py:905] [0/1] [4bryyl4ahh5whyg3zwqebpwmjnx6w77nqgqbdjlowju6lkqtn7w] inductor_config[post_grad_fusion_options]: {}
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[reorder_for_locality]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[dynamic_scale_rblock]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_fuse_int_mm_with_mul]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[use_mixed_mm]: True
torch/_inductor/codecache.py:905] [0/1] [zwmmbkdkarexuhbigurz5lfnhx64tht7fznecjkrvznh6rzivbv] inductor_config[fx_passes_numeric_check]: {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch/_inductor/codecache.py:905] [0/1] [v2td5s4lnsvyxvaevy4chx6kc5h3mm2axazbgwimqule5zrzao7] inductor_config[mixed_mm_choice]: heuristic
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[reorder_for_compute_comm_overlap]: False
torch/_inductor/codecache.py:905] [0/1] [ssupi7bu3rrhdpg2jyegzncu3kg3nnhklyliqvutaxgs7y7k3dx] inductor_config[reorder_for_compute_comm_overlap_passes]: ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[reorder_for_peak_memory]: True
torch/_inductor/codecache.py:905] [0/1] [lxxtoqhcoepwfokeiibd575gnxo3uzwiv4hmpomlwkpzqz3qzsh] inductor_config[estimate_op_runtime]: default
torch/_inductor/codecache.py:905] [0/1] [yezuzjtg4h3jjur4jwtwiehbyixa7eonq4tqsqmwqve2lvvmrem] inductor_config[intra_node_bw]: 300
torch/_inductor/codecache.py:905] [0/1] [5fxczt3ciyxitdhizb7sfsgn7fhpczcqsngttnt5ot2wyctk7co] inductor_config[inter_node_bw]: 25
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[max_autotune]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[max_autotune_pointwise]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[max_autotune_gemm]: False
torch/_inductor/codecache.py:905] [0/1] [j6c55jha5r2sdys2rwq7uqhtleea5dgjcye7nicfgft36v7xfvp] inductor_config[autotune_num_choices_displayed]: 10
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_same_precision]: False
torch/_inductor/codecache.py:905] [0/1] [2y7luesktjrque3nr7qtxnum2mkbeegzdrsvkm3rvdlhqboajhx] inductor_config[max_autotune_gemm_backends]: ATEN,TRITON,CPP
torch/_inductor/codecache.py:905] [0/1] [uqlsbif4zxd75vt522p52txyuguieipi2lwz5g5awt56lccqk7s] inductor_config[max_autotune_conv_backends]: ATEN,TRITON
torch/_inductor/codecache.py:905] [0/1] [jvchmi66fvqzlemhr5fcqorz5trfdtdalzfagtj2aolmimwqhdq] inductor_config[max_autotune_gemm_search_space]: DEFAULT
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[autotune_fallback_to_aten]: True
torch/_inductor/codecache.py:905] [0/1] [wft6ljqsfr3x4m7fa5zuyb7cwknky4irrxz4bjr6uzr2yiopxqj] inductor_config[unbacked_symint_fallback]: 8192
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[search_autotune_cache]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[save_args]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[autotune_in_subproc]: False
torch/_inductor/codecache.py:905] [0/1] [iglov24t7x5ruci344aer2tm6nqshi4veuw4wxlssxtu46cx76m] inductor_config[max_autotune_subproc_result_timeout_seconds]: 60.0
torch/_inductor/codecache.py:905] [0/1] [bh33ranllcgilhgmgr3qvygzxjm6isq5iexnfm3zx6fnr2zwlp2] inductor_config[max_autotune_subproc_graceful_timeout_seconds]: 1.0
torch/_inductor/codecache.py:905] [0/1] [pwoh5aypf4fxbntdvwt67rppxorqos6xr3w7qzeun6kblbfg2ga] inductor_config[max_autotune_subproc_terminate_timeout_seconds]: 2.0
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[autotune_multi_device]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[coordinate_descent_tuning]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[coordinate_descent_check_all_directions]: False
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[coordinate_descent_search_radius]: 1
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[autoheuristic_collect]:
torch/_inductor/codecache.py:905] [0/1] [jwbrgxes7vjqumngs5hyj6gn5nytv2whnppnzngvaagfmawhkkd] inductor_config[autoheuristic_use]: mixed_mm
torch/_inductor/codecache.py:905] [0/1] [jvchmi66fvqzlemhr5fcqorz5trfdtdalzfagtj2aolmimwqhdq] inductor_config[autoheuristic_log_path]: DEFAULT
torch/_inductor/codecache.py:905] [0/1] [4p2fdjlvxrcw7c7fvzm5huhtqxnro4kvkx56f7p5zyrxqkwooov] inductor_config[layout_opt_default]: 1
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[layout_optimization]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_layout_optimization]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[keep_output_stride]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[warn_mix_layout]: False
torch/_inductor/codecache.py:905] [0/1] [lkkae3meylaixfif4thncru4hjqeaislawjoghffrbwuscaagei] inductor_config[realize_reads_threshold]: 4
torch/_inductor/codecache.py:905] [0/1] [rr5m5hsocoyodldz7vcvaizdwvm2rt34evmqdxvng7wz3tufvo6] inductor_config[realize_opcount_threshold]: 30
torch/_inductor/codecache.py:905] [0/1] [yttmfmxblgcbsvbokguzowcorrcxz5uunxtcvsbe6nijgcx45he] inductor_config[realize_acc_reads_threshold]: 8
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[fallback_random]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[implicit_fallbacks]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aggressive_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[debug_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[benchmark_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[enabled_metric_tables]:
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[loop_ordering_after_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [j6c55jha5r2sdys2rwq7uqhtleea5dgjcye7nicfgft36v7xfvp] inductor_config[score_fusion_memory_threshold]: 10
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[benchmark_epilogue_fusion]: True
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[max_epilogue_benchmarked_choices]: 1
torch/_inductor/codecache.py:905] [0/1] [jykiys6ynafs3zdylwa5ggq6j655mxeh42d6mtdi22gffkrmiac] inductor_config[max_fusion_size]: 64
torch/_inductor/codecache.py:905] [0/1] [yttmfmxblgcbsvbokguzowcorrcxz5uunxtcvsbe6nijgcx45he] inductor_config[max_pointwise_cat_inputs]: 8
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_pointwise_cat]: False
torch/_inductor/codecache.py:905] [0/1] [yttmfmxblgcbsvbokguzowcorrcxz5uunxtcvsbe6nijgcx45he] inductor_config[unroll_reductions_threshold]: 8
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[comment_origin]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[conv_1x1_as_mm]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[split_reductions]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[benchmark_kernel]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[constant_and_index_propagation]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[always_keep_tensor_constants]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[assert_indirect_indexing]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[compute_all_bounds]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[combo_kernels]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[benchmark_combo_kernel]: False
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[combo_kernels_autotune]: 1
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[combo_kernel_allow_mixed_sizes]: 1
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[combo_kernel_foreach_dynamic_shapes]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[joint_graph_constant_folding]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[debug_index_asserts]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[emulate_precision_casts]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[is_nightly_or_source]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[developer_warnings]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[optimize_scatter_upon_const_tensor]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[global_cache_dir]: None
torch/_inductor/codecache.py:905] [0/1] [j6c55jha5r2sdys2rwq7uqhtleea5dgjcye7nicfgft36v7xfvp] inductor_config[kernel_name_max_ops]: 10
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[shape_padding]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[comprehensive_padding]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[pad_channels_last]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[disable_padding_cpu]: True
torch/_inductor/codecache.py:905] [0/1] [ljdqgtysl3vdf7j6attlz5gmjg2ncihnveojfyubosplmkrjgra] inductor_config[padding_alignment_bytes]: 128
torch/_inductor/codecache.py:905] [0/1] [dnnw5ks3yxrp7mwvihb2hh4tqx35ye637xt33x64kw4fvz2nyzg] inductor_config[padding_stride_threshold]: 1024
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[pad_outputs]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[bw_outputs_user_visible]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[force_shape_pad]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[permute_fusion]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[profiler_mark_wrapper_call]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[generate_intermediate_hooks]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[debug_ir_traceback]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[profile_bandwidth]: False
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[profile_bandwidth_regex]:
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[profile_bandwidth_output]: None
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[profile_bandwidth_with_do_bench_using_profiling]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[disable_cpp_codegen]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[freezing]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[freezing_discard_parameters]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[decompose_mem_bound_mm]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[assume_aligned_inputs]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[unsafe_ignore_unsupported_triton_autotune_args]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[check_stack_no_cycles_TESTING_ONLY]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[enable_linear_binary_folding]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[annotate_training]: False
torch/_inductor/codecache.py:905] [0/1] [sz3im5ogc6asp7g4uqocnovype63tkdexzfrniv6hn2oank3biu] inductor_config[cpp.threads]: -1
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[cpp.no_redundant_loops]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp.dynamic_threads]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.simdlen]: None
torch/_inductor/codecache.py:905] [0/1] [g7rrnbg5yonzux3cfj5ovre5lob3ayda7qcfpxjvtwmiz4uicii] inductor_config[cpp.min_chunk_size]: 4096
torch/_inductor/codecache.py:905] [0/1] [c7zj4qytmety6keurs3hsh5wn7foxp3dqx4kym2ucszzcb2ngrf] inductor_config[cpp.cxx]: (None, 'g++')
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp.enable_kernel_profile]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[cpp.weight_prepack]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.inject_relu_bug_TESTING_ONLY]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.inject_log1p_bug_TESTING_ONLY]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.vec_isa_ok]: None
torch/_inductor/codecache.py:905] [0/1] [yrty22bseefglnysuoec4ji7j2rnaggdj3g33zzj7avogwfmgdw] inductor_config[cpp.descriptive_names]: original_aten
torch/_inductor/codecache.py:905] [0/1] [ebt2ncs4f5y7dn7btzi76mnouepvzad474tmp5iju4wiuumjl4s] inductor_config[cpp.max_horizontal_fusion_size]: 16
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[cpp.fallback_scatter_reduce_sum]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp.enable_unsafe_math_opt_flag]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp.enable_floating_point_contract_flag]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[cpp.enable_tiling_heuristics]: True
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[cpp.gemm_max_k_slices]: 1
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.gemm_cache_blocking]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cpp.gemm_thread_factors]: None
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[cpp.enable_loop_tail_vec]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cpp.enable_concat_linear]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.cudagraphs]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.cudagraph_trees]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.cudagraph_skip_dynamic_graphs]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.slow_path_cudagraph_asserts]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.cudagraph_trees_history_recording]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.cudagraph_support_input_mutation]: True
torch/_inductor/codecache.py:905] [0/1] [ljdqgtysl3vdf7j6attlz5gmjg2ncihnveojfyubosplmkrjgra] inductor_config[triton.cudagraph_unexpected_rerecord_limit]: 128
torch/_inductor/codecache.py:905] [0/1] [tuax46wac7rfv2trf5gcps6vleo3cq44lbnrdxtprvo3ljjaddj] inductor_config[triton.cudagraph_dynamic_shape_warn_limit]: 50
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.force_cudagraph_sync]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.force_cudagraphs_warmup]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.fast_path_cudagraph_asserts]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.skip_cudagraph_warmup]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.debug_sync_graph]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.debug_sync_kernel]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.dense_indexing]: False
torch/_inductor/codecache.py:905] [0/1] [pr5nr4a7dthirgd2ljo3d2xakc63ywxugusu6mkmr6gmpeliyib] inductor_config[triton.max_tiles]: 2
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.prefer_nd_tiling]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.autotune_pointwise]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.autotune_cublasLt]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[triton.autotune_at_compile_time]: None
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.tiling_prevents_pointwise_fusion]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.tiling_prevents_reduction_fusion]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.unique_kernel_names]: True
torch/_inductor/codecache.py:905] [0/1] [yrty22bseefglnysuoec4ji7j2rnaggdj3g33zzj7avogwfmgdw] inductor_config[triton.descriptive_names]: original_aten
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.persistent_reductions]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.cooperative_reductions]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.force_cooperative_reductions]: False
torch/_inductor/codecache.py:905] [0/1] [vrl5ktomgtzox5xucd3np6vug3vyj6hwwzahqijuwpmamlv7ohi] inductor_config[triton.multi_kernel]: 0
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.divisible_by_16]: True
torch/_inductor/codecache.py:905] [0/1] [fv6slhtedtydps5s5u2etitscliblzcidyitqf7krsv4e23fzk6] inductor_config[triton.min_split_scan_rblock]: 256
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.store_cubin]: False
torch/_inductor/codecache.py:905] [0/1] [ebt2ncs4f5y7dn7btzi76mnouepvzad474tmp5iju4wiuumjl4s] inductor_config[triton.spill_threshold]: 16
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.use_block_ptr]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[triton.inject_relu_bug_TESTING_ONLY]: None
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[triton.codegen_upcast_to_fp32]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[triton.enable_persistent_tma_matmul]: False
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[aot_inductor.output_path]:
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.debug_compile]: False
torch/_inductor/codecache.py:905] [0/1] [ngkkx5e6z7erl6da23zb2cmsctz4yvaqyameyg5hbqln4wrhh7x] inductor_config[aot_inductor.debug_intermediate_value_printer]: 0
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[aot_inductor.filtered_kernel_names]: None
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[aot_inductor.serialized_in_spec]:
torch/_inductor/codecache.py:905] [0/1] [v3hzzlv4tjgvp3pyhmzagjd25orl6n7nynoa7svlhhwk73b7u3c] inductor_config[aot_inductor.serialized_out_spec]:
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.use_runtime_constant_folding]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.force_mmap_weights]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.package]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.package_cpp_only]: False
torch/_inductor/codecache.py:905] [0/1] [4bryyl4ahh5whyg3zwqebpwmjnx6w77nqgqbdjlowju6lkqtn7w] inductor_config[aot_inductor.metadata]: {}
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[aot_inductor.raise_error_on_ignored_optimization]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.dump_aoti_minifier]: False
torch/_inductor/codecache.py:905] [0/1] [4bryyl4ahh5whyg3zwqebpwmjnx6w77nqgqbdjlowju6lkqtn7w] inductor_config[aot_inductor.presets]: {}
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.allow_stack_allocation]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[aot_inductor.use_minimal_arrayref_interface]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[aot_inductor.package_constants_in_so]: True
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cuda.arch]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cuda.version]: None
torch/_inductor/codecache.py:905] [0/1] [tvyftmtdmezlejo2xllu7awzv4pzc4vm4fub4b3gpl5jptjkosi] inductor_config[cuda.compile_opt_level]: -O1
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cuda.enable_cuda_lto]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cuda.enable_ptxas_info]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cuda.enable_debug_info]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cuda.use_fast_math]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cuda.cutlass_max_profiling_configs]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cuda.cuda_cxx]: None
torch/_inductor/codecache.py:905] [0/1] [aghvyrrgwvxijco2pk5wzc3cgmmthrbmgxitiibxuuscxdwrjd3] inductor_config[cuda.cutlass_backend_min_gemm_size]: 1
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[cuda.generate_test_runner]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[cuda.cutlass_op_allowlist_regex]: None
torch/_inductor/codecache.py:905] [0/1] [lwkz5chtpji756gurqw4foijfi7zfgljtnn5nmnvdi2skpt4mgh] inductor_config[cuda.cutlass_op_denylist_regex]: pingpong
torch/_inductor/codecache.py:905] [0/1] [h25wqx6vliw4j5rtzzbv6latydxyei3deyg6v7wzvnzryfktuki] inductor_config[rocm.arch]: []
torch/_inductor/codecache.py:905] [0/1] [oartxnko2l7d67tzwwm2otcumaut3n4wwcfgz3o377hmcveu5ft] inductor_config[rocm.ck_supported_arch]: ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch/_inductor/codecache.py:905] [0/1] [klfqjprnpfhcdurgvuikvc4rpd5ynkpk77toousr5h3u5roty6p] inductor_config[rocm.compile_opt_level]: -O2
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[rocm.is_debug]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[rocm.save_temps]: False
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[rocm.use_fast_math]: True
torch/_inductor/codecache.py:905] [0/1] [cev5uo2jlwdhw2uyzcm7vr6cl23azjfw437f5r5lskm7spucos6] inductor_config[rocm.flush_denormals]: True
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[rocm.print_kernel_resource_usage]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[rocm.rocm_home]: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[rocm.ck_dir]: None
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[rocm.generate_test_runner]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] inductor_config[rocm.n_max_profiling_configs]: None
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[rocm.use_preselected_instances]: False
torch/_inductor/codecache.py:905] [0/1] [bsvfcwwoczx2rlkdz2eta6doujsymyihmi46hhwk6clrrvwcb6m] inductor_config[cpu_backend]: cpp
torch/_inductor/codecache.py:905] [0/1] [caw4ly2z672k6kjfahoxwpajp5idhhtrpgf3ma2clylcp7c7aid] inductor_config[cuda_backend]: triton
torch/_inductor/codecache.py:905] [0/1] [ljhgflgihidopsfsdcbqynv27nceykby3nutyd5jlcpq7n6e7l4] inductor_config[halide.cpu_target]: host
torch/_inductor/codecache.py:905] [0/1] [wx7vmsmrdpk5ue2txlywp3lj3faqmdjphs5fgg2ehzsyno7uovg] inductor_config[halide.gpu_target]: host-cuda
torch/_inductor/codecache.py:905] [0/1] [svgytlua5wcyeia7wq7e6zgh5tsueikrnzchmdmouvmkpfsc2zq] inductor_config[halide.scheduler_cuda]: Anderson2021
torch/_inductor/codecache.py:905] [0/1] [k5ogk6345jvklsnu7g2njqstiz2g6pm5wmqpgg3kasrmuqwjvl6] inductor_config[halide.scheduler_cpu]: Adams2019
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[halide.asserts]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[halide.debug]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[halide.scan_kernels]: False
torch/_inductor/codecache.py:905] [0/1] [h25wqx6vliw4j5rtzzbv6latydxyei3deyg6v7wzvnzryfktuki] inductor_config[external_matmul]: []
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[test_configs.force_extern_kernel_in_multi_template]: False
torch/_inductor/codecache.py:905] [0/1] [esstihe2nyydk4mhzpvox3qkajyu5y5t23hk3fi2me7jn75xi3o] inductor_config[test_configs.runtime_triton_dtype_assert]: False
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] post_grad_custom_pre_pass: None
torch/_inductor/codecache.py:905] [0/1] [tquy2we2efmowuj4wuqzcfcfdcrkzkzmwdae6hprj7fa64jpusq] post_grad_custom_post_pass: None
torch/_inductor/codecache.py:1348] [0/1] fx graph cache miss for key ff5tzdurkhhdldzxihywjaapklrfql4ssnonbev5cdxzg3qfycqf
torch/_inductor/triton_bundler.py:110] [0/1] TritonBundler.begin_compile is called
torch/_inductor/compile_fx.py:843] [0/1] Step 3: torchinductor compiling FORWARDS graph 0
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs] TRACED GRAPH
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]  ===== AFTER POST GRAD =====
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]  /home/mattarde/.local/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class <lambda>(torch.nn.Module):
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]     def forward(self, arg0_1: "f32[8, 16][16, 1]cpu", arg1_1: "f32[16][1]cpu"):
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]          # File: /mnt/c/work/pytorch_debug/kernel.py:21 in simple_kernel, code: y = x * w               # broadcast mul
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         mul: "f32[8, 16][16, 1]cpu" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]          # File: /mnt/c/work/pytorch_debug/kernel.py:22 in simple_kernel, code: y = torch.relu(y)       # pointwise
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         relu: "f32[8, 16][16, 1]cpu" = torch.ops.aten.relu.default(mul);  mul = None
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]          # File: /mnt/c/work/pytorch_debug/kernel.py:23 in simple_kernel, code: s = y.sum(dim=1)        # reduction
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         sum_1: "f32[8][1]cpu" = torch.ops.aten.sum.dim_IntList(relu, [1]);  relu = None
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]          # File: /mnt/c/work/pytorch_debug/kernel.py:24 in simple_kernel, code: out = s * 0.1 + 1.0     # pointwise
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         mul_1: "f32[8][1]cpu" = torch.ops.aten.mul.Tensor(sum_1, 0.1);  sum_1 = None
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         add: "f32[8][1]cpu" = torch.ops.aten.add.Tensor(mul_1, 1.0);  mul_1 = None
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]         return (add,)
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]
torch/_inductor/compile_fx.py:910] [0/1] [__post_grad_graphs]
torch/_inductor/graph.py:1433] [0/1] lowering %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
torch/_inductor/graph.py:1433] [0/1] lowering %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
torch/_inductor/graph.py:1433] [0/1] lowering %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
torch/_inductor/graph.py:1125] [0/1]   via <function mul at 0x79b1f1a14d30>
torch/_inductor/graph.py:1433] [0/1] lowering %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mul,), kwargs = {})
torch/_inductor/graph.py:1125] [0/1]   via <function make_pointwise.<locals>.inner at 0x79b1f1a17490>
torch/_inductor/graph.py:1433] [0/1] lowering %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%relu, [1]), kwargs = {})
torch/_inductor/graph.py:1125] [0/1]   via <function sum_ at 0x79b1f1a15240>
torch/_inductor/graph.py:1433] [0/1] lowering %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.1), kwargs = {})
torch/_inductor/graph.py:1125] [0/1]   via <function mul at 0x79b1f1a14d30>
torch/_inductor/graph.py:1433] [0/1] lowering %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 1.0), kwargs = {})
torch/_inductor/graph.py:1125] [0/1]   via <function make_pointwise.<locals>.inner at 0x79b1f1a16710>
torch/_inductor/graph.py:1433] [0/1] lowering return (add,)
torch/_inductor/graph.py:1278] [0/1] Force channels last inputs for 0 conv for the current graph with id 0
torch/_inductor/scheduler.py:2081] [0/1] scheduling ComputedBuffer(name='buf0', layout=FixedLayout('cpu', torch.float32, size=[8], stride=[1]), data=Reduction(
torch/_inductor/scheduler.py:2081] [0/1]   'cpu',
torch/_inductor/scheduler.py:2081] [0/1]   torch.float32,
torch/_inductor/scheduler.py:2081] [0/1]   def inner_fn(index, rindex):
torch/_inductor/scheduler.py:2081] [0/1]       i0 = index
torch/_inductor/scheduler.py:2081] [0/1]       r0 = rindex
torch/_inductor/scheduler.py:2081] [0/1]       tmp0 = ops.load(arg0_1, r0 + 16 * i0)
torch/_inductor/scheduler.py:2081] [0/1]       tmp1 = ops.load(arg1_1, r0)
torch/_inductor/scheduler.py:2081] [0/1]       tmp2 = tmp0 * tmp1
torch/_inductor/scheduler.py:2081] [0/1]       tmp3 = ops.relu(tmp2)
torch/_inductor/scheduler.py:2081] [0/1]       return tmp3
torch/_inductor/scheduler.py:2081] [0/1]   ,
torch/_inductor/scheduler.py:2081] [0/1]   ranges=[8],
torch/_inductor/scheduler.py:2081] [0/1]   reduction_ranges=[16],
torch/_inductor/scheduler.py:2081] [0/1]   reduction_type=sum,
torch/_inductor/scheduler.py:2081] [0/1]   origin_node=sum_1,
torch/_inductor/scheduler.py:2081] [0/1]   origins=OrderedSet([mul, sum_1, relu])
torch/_inductor/scheduler.py:2081] [0/1] ))
torch/_inductor/scheduler.py:2081] [0/1] scheduling ComputedBuffer(name='buf1', layout=FixedLayout('cpu', torch.float32, size=[8], stride=[1]), data=Pointwise(device=device(type='cpu'), dtype=torch.float32, inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x79b1e91e5fc0>, ranges=[8]))
torch/_inductor/scheduler.py:2158] [0/1] scheduling output buf1
torch/_inductor/memory.py:602] [0/1] Reordering for peak memory -- 1 nodes
torch/_inductor/memory.py:630] [0/1] Baseline peak memory: 64
torch/_inductor/memory.py:648] [0/1] topological_sort_lpmf peak memory: 64
torch/_inductor/memory.py:648] [0/1] topological_sort_bfs peak memory: 64
torch/_inductor/memory.py:648] [0/1] topological_sort_dfs peak memory: 64
torch/_inductor/scheduler.py:3504] [0/1] Generating code for node op0_op1 with estimated runtime 0.000000
torch/_inductor/bounds.py:72] [0/1] get_bounds:
torch/_inductor/bounds.py:72] [0/1] graph():
torch/_inductor/bounds.py:72] [0/1]     %ops : [num_users=6] = placeholder[target=ops]
torch/_inductor/bounds.py:72] [0/1]     %get_index : [num_users=1] = call_module[target=get_index](args = (index0,), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %load : [num_users=1] = call_method[target=load](args = (%ops, arg0_1, %get_index), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %get_index_1 : [num_users=1] = call_module[target=get_index](args = (index1,), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %load_1 : [num_users=1] = call_method[target=load](args = (%ops, arg1_1, %get_index_1), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %mul : [num_users=1] = call_method[target=mul](args = (%ops, %load, %load_1), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %relu : [num_users=1] = call_method[target=relu](args = (%ops, %mul), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %reduction : [num_users=1] = call_method[target=reduction](args = (%ops, torch.float32, torch.float32, sum, %relu), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %get_index_2 : [num_users=1] = call_module[target=get_index](args = (index2,), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %store_reduction : [num_users=1] = call_method[target=store_reduction](args = (%ops, buf0, %get_index_2, %reduction), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     return store_reduction
torch/_inductor/bounds.py:72] [0/1] get_bounds:
torch/_inductor/bounds.py:72] [0/1] graph():
torch/_inductor/bounds.py:72] [0/1]     %ops : [num_users=6] = placeholder[target=ops]
torch/_inductor/bounds.py:72] [0/1]     %get_index : [num_users=1] = call_module[target=get_index](args = (index0,), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %load : [num_users=1] = call_method[target=load](args = (%ops, buf0, %get_index), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %constant : [num_users=1] = call_method[target=constant](args = (%ops, 0.1, torch.float32), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %mul : [num_users=1] = call_method[target=mul](args = (%ops, %load, %constant), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %constant_1 : [num_users=1] = call_method[target=constant](args = (%ops, 1.0, torch.float32), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %add : [num_users=1] = call_method[target=add](args = (%ops, %mul, %constant_1), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %get_index_1 : [num_users=1] = call_module[target=get_index](args = (index0,), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     %store : [num_users=1] = call_method[target=store](args = (%ops, buf1, %get_index_1, %add, None), kwargs = {})
torch/_inductor/bounds.py:72] [0/1]     return store
torch/_inductor/graph.py:1970] [0/1] Finished codegen for all nodes. The list of kernel names available: OrderedSet([])
torch/_inductor/graph.py:2045] [0/1] [__output_code] Output code:
torch/_inductor/graph.py:2045] [0/1] [__output_code] # AOT ID: ['0_inference']
torch/_inductor/graph.py:2045] [0/1] [__output_code] from ctypes import c_void_p, c_long, c_int
torch/_inductor/graph.py:2045] [0/1] [__output_code] import torch
torch/_inductor/graph.py:2045] [0/1] [__output_code] import math
torch/_inductor/graph.py:2045] [0/1] [__output_code] import random
torch/_inductor/graph.py:2045] [0/1] [__output_code] import os
torch/_inductor/graph.py:2045] [0/1] [__output_code] import tempfile
torch/_inductor/graph.py:2045] [0/1] [__output_code] from math import inf, nan
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.hooks import run_intermediate_hooks
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.utils import maybe_profile
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.codegen.memory_planning import _align as align
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch import device, empty_strided
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.async_compile import AsyncCompile
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.select_algorithm import extern_kernels
torch/_inductor/graph.py:2045] [0/1] [__output_code] from torch._inductor.codegen.multi_kernel import MultiKernelCall
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] aten = torch.ops.aten
torch/_inductor/graph.py:2045] [0/1] [__output_code] inductor_ops = torch.ops.inductor
torch/_inductor/graph.py:2045] [0/1] [__output_code] _quantized = torch.ops._quantized
torch/_inductor/graph.py:2045] [0/1] [__output_code] assert_size_stride = torch._C._dynamo.guards.assert_size_stride
torch/_inductor/graph.py:2045] [0/1] [__output_code] empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
torch/_inductor/graph.py:2045] [0/1] [__output_code] empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
torch/_inductor/graph.py:2045] [0/1] [__output_code] empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
torch/_inductor/graph.py:2045] [0/1] [__output_code] reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
torch/_inductor/graph.py:2045] [0/1] [__output_code] alloc_from_pool = torch.ops.inductor._alloc_from_pool
torch/_inductor/graph.py:2045] [0/1] [__output_code] async_compile = AsyncCompile()
torch/_inductor/graph.py:2045] [0/1] [__output_code] empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] cpp_fused_add_mul_relu_sum_0 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*'], '''
torch/_inductor/graph.py:2045] [0/1] [__output_code] #include "/mnt/c/work/pytorch_debug/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
torch/_inductor/graph.py:2045] [0/1] [__output_code] extern "C"  void kernel(float* in_out_ptr0,
torch/_inductor/graph.py:2045] [0/1] [__output_code]                        const float* in_ptr0,
torch/_inductor/graph.py:2045] [0/1] [__output_code]                        const float* in_ptr1)
torch/_inductor/graph.py:2045] [0/1] [__output_code] {
torch/_inductor/graph.py:2045] [0/1] [__output_code]     auto out_ptr0 = in_out_ptr0;
torch/_inductor/graph.py:2045] [0/1] [__output_code]     {
torch/_inductor/graph.py:2045] [0/1] [__output_code]         #pragma GCC ivdep
torch/_inductor/graph.py:2045] [0/1] [__output_code]         for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
torch/_inductor/graph.py:2045] [0/1] [__output_code]         {
torch/_inductor/graph.py:2045] [0/1] [__output_code]             {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 float tmp_acc0 = 0;
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(16L); x1+=static_cast<int64_t>(8L))
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                         if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16L)))
torch/_inductor/graph.py:2045] [0/1] [__output_code]                         {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                             auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 16L*x0), static_cast<int64_t>(8));
torch/_inductor/graph.py:2045] [0/1] [__output_code]                             auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
torch/_inductor/graph.py:2045] [0/1] [__output_code]                             auto tmp2 = tmp0 * tmp1;
torch/_inductor/graph.py:2045] [0/1] [__output_code]                             auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
torch/_inductor/graph.py:2045] [0/1] [__output_code]                             tmp_acc0_vec = tmp_acc0_vec + tmp3;
torch/_inductor/graph.py:2045] [0/1] [__output_code]                         }
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     }
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 }
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 in_out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
torch/_inductor/graph.py:2045] [0/1] [__output_code]             }
torch/_inductor/graph.py:2045] [0/1] [__output_code]         }
torch/_inductor/graph.py:2045] [0/1] [__output_code]     }
torch/_inductor/graph.py:2045] [0/1] [__output_code]     {
torch/_inductor/graph.py:2045] [0/1] [__output_code]         for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
torch/_inductor/graph.py:2045] [0/1] [__output_code]         {
torch/_inductor/graph.py:2045] [0/1] [__output_code]             {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 {
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp1 = static_cast<float>(0.1);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp2 = at::vec::Vectorized<float>(tmp1);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp3 = tmp0 * tmp2;
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp4 = static_cast<float>(1.0);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp5 = at::vec::Vectorized<float>(tmp4);
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     auto tmp6 = tmp3 + tmp5;
torch/_inductor/graph.py:2045] [0/1] [__output_code]                     tmp6.store(in_out_ptr0 + static_cast<int64_t>(x0));
torch/_inductor/graph.py:2045] [0/1] [__output_code]                 }
torch/_inductor/graph.py:2045] [0/1] [__output_code]             }
torch/_inductor/graph.py:2045] [0/1] [__output_code]         }
torch/_inductor/graph.py:2045] [0/1] [__output_code]     }
torch/_inductor/graph.py:2045] [0/1] [__output_code] }
torch/_inductor/graph.py:2045] [0/1] [__output_code] ''')
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] async_compile.wait(globals())
torch/_inductor/graph.py:2045] [0/1] [__output_code] del async_compile
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] def call(args):
torch/_inductor/graph.py:2045] [0/1] [__output_code]     arg0_1, arg1_1 = args
torch/_inductor/graph.py:2045] [0/1] [__output_code]     args.clear()
torch/_inductor/graph.py:2045] [0/1] [__output_code]     assert_size_stride(arg0_1, (8, 16), (16, 1))
torch/_inductor/graph.py:2045] [0/1] [__output_code]     assert_size_stride(arg1_1, (16, ), (1, ))
torch/_inductor/graph.py:2045] [0/1] [__output_code]     buf0 = empty_strided_cpu((8, ), (1, ), torch.float32)
torch/_inductor/graph.py:2045] [0/1] [__output_code]     buf1 = buf0; del buf0  # reuse
torch/_inductor/graph.py:2045] [0/1] [__output_code]     cpp_fused_add_mul_relu_sum_0(buf1, arg0_1, arg1_1)
torch/_inductor/graph.py:2045] [0/1] [__output_code]     del arg0_1
torch/_inductor/graph.py:2045] [0/1] [__output_code]     del arg1_1
torch/_inductor/graph.py:2045] [0/1] [__output_code]     return (buf1, )
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] def benchmark_compiled_module(times=10, repeat=10):
torch/_inductor/graph.py:2045] [0/1] [__output_code]     from torch._dynamo.testing import rand_strided
torch/_inductor/graph.py:2045] [0/1] [__output_code]     from torch._inductor.utils import print_performance
torch/_inductor/graph.py:2045] [0/1] [__output_code]     arg0_1 = rand_strided((8, 16), (16, 1), device='cpu', dtype=torch.float32)
torch/_inductor/graph.py:2045] [0/1] [__output_code]     arg1_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
torch/_inductor/graph.py:2045] [0/1] [__output_code]     fn = lambda: call([arg0_1, arg1_1])
torch/_inductor/graph.py:2045] [0/1] [__output_code]     return print_performance(fn, times=times, repeat=repeat)
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2045] [0/1] [__output_code] if __name__ == "__main__":
torch/_inductor/graph.py:2045] [0/1] [__output_code]     from torch._inductor.wrapper_benchmark import compiled_module_main
torch/_inductor/graph.py:2045] [0/1] [__output_code]     compiled_module_main('None', benchmark_compiled_module)
torch/_inductor/graph.py:2045] [0/1] [__output_code]
torch/_inductor/graph.py:2053] [0/1] [__output_code] Output code written to: /mnt/c/work/pytorch_debug/vt/cvtee6a3paljsjblrayj6nvwutwqcz6iro7awbwmknpajuryde24.py
torch/_inductor/graph.py:2086] [0/1] Output code written to: /mnt/c/work/pytorch_debug/vt/cvtee6a3paljsjblrayj6nvwutwqcz6iro7awbwmknpajuryde24.py
torch/_inductor/graph.py:2087] [0/1] [__output_code] Output code written to: /mnt/c/work/pytorch_debug/vt/cvtee6a3paljsjblrayj6nvwutwqcz6iro7awbwmknpajuryde24.py
torch/_inductor/triton_bundler.py:120] [0/1] TritonBundler.end_compile is called
torch/_inductor/triton_bundler.py:120] [0/1] TritonBundler.end_compile is called
torch/_inductor/compile_fx.py:768] [0/1] FX codegen and compilation took 7.934s
torch/_inductor/compile_fx.py:770] [0/1] Step 3: torchinductor done compiling FORWARDS graph 0
torch/_dynamo/variables/tensor.py:869] [1/0] Graph break from `Tensor.item()`, consider setting:
torch/_dynamo/variables/tensor.py:869] [1/0]     torch._dynamo.config.capture_scalar_outputs = True
torch/_dynamo/variables/tensor.py:869] [1/0] or:
torch/_dynamo/variables/tensor.py:869] [1/0]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
torch/_dynamo/variables/tensor.py:869] [1/0] to include these operations in the captured graph.
torch/_dynamo/variables/tensor.py:869] [1/0]
torch/_dynamo/variables/tensor.py:869] [1/0] Graph break: from user code at:
torch/_dynamo/variables/tensor.py:869] [1/0]   File "/mnt/c/work/pytorch_debug/kernel.py", line 50, in kernel_with_graph_break
torch/_dynamo/variables/tensor.py:869] [1/0]     if y[0].item() > 0:
torch/_dynamo/variables/tensor.py:869] [1/0]
torch/_dynamo/variables/tensor.py:869] [1/0]
torch/_inductor/remote_cache.py:414] Cache Metrics: None
torch/_inductor/remote_cache.py:414]
```
