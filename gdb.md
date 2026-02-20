

### Backstrace on function
```
set pagination off
set logging file gdb.output
set logging on
# This line is a comment
break function_name
    command 1
    backtrace full
    continue
end
run
set logging off
quit
```
### SO plugin load events, if dlopens
```
set pagination off
set breakpoint pending on
set print thread-events off
set detach-on-fork off
set follow-fork-mode parent
set stop-on-solib-events 1

break dlopen
commands
  silent
  printf "dlopen(%s)\n", (char*)$rdi   # x86_64: arg1 in RDI
  continue
end
```
### Debug until value is
+ https://github.com/slamko/gdb-debug-until
+ https://github.com/slamko/gdb-debug-until/blob/master/examples.md
### GDB Command 
```
class PP(gdb.Command):
  """print value history index, name, and value of each arg"""

  def __init__(self):
    super(PP, self).__init__("pp", gdb.COMMAND_DATA, gdb.COMPLETE_EXPRESSION)

  def invoke(self, argstr, from_tty):
    for arg in gdb.string_to_argv(argstr):
      line = gdb.execute("print " + arg, from_tty=False, to_string=True)
      line = line.replace("=", "= " + arg + " =", 1)
      gdb.write(line)

PP()
```

### Show All input args
```
define p
  set confirm off
  eval "undisplay"
  set confirm on
  set $i = 0
  while $i < $argc
    eval "display $arg%d", $i
    set $i = $i + 1
  end
  display
end
```

## GEF
### https://github.com/rustymagnet3000/gdb#memory

## LLDB
### https://github.com/rustymagnet3000/lldb_debugger

## Bash and GDB
### Wait for Process to start
```
#!/bin/sh
programname="$1"
delay=0.5
while true
do
   pid=ps -C "${programname}" -o pid=
if [ $? -eq 0 ]
then
   echo ${pid}
exit
else
   sleep ${delay}
fi
done
```

## C/ C++ codes
1. https://nullprogram.com/blog/2024/01/28/
```
#define assert(c)  while (!(c)) __builtin_trap()
#define assert(c)  while (!(c)) __builtin_unreachable()
#define assert(c)  while (!(c)) *(volatile int *)0 = 0
#define assert(c)  do if (!(c)) __debugbreak(); while (0)
#define breakpoint()  asm ("int3; nop")
```
