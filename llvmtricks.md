# lit
+ Run lit test if target is registered
  `; RUN: %if x86-registered-target %{ some-llvm-tool ... %}`
+ Optimize prefixes by combining them
  
  ` ;RUN: <> | Filecheck --check=prefixes=GENERIC,SPECIAL1`
  ` ;RUN: <> | Filecheck --check=prefixes=GENERIC,SPECIAL2`
  
