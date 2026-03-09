# Linux settings 

## Customizable version of GNU parallel
+ https://github.com/mahesh-attarde/groot/blob/main/pyscripts/parallel_exec.py
```
    # Compile multiple files in parallel
    parallel_exec.py compile_commands.txt -j 8

    # Run test suite with first test serial, rest parallel
    parallel_exec.py test_commands.txt -s 2 -j 16

    # Process batch of commands, save to custom log, quiet mode
    parallel_exec.py batch_process.txt -j 32 -o results.log -q

    # Execute subset of long command list for debugging
    parallel_exec.py all_commands.txt --range 45:50 -vv --keep-temp

    # Run critical commands serially, stop on first failure
    parallel_exec.py critical_ops.txt -s 1 --halt-on-error
```
## Increase Resources
`ulimit -s unlimited ; ulimit -n 2048`
