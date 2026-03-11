# Linux commands
```
dot -Tpng temp.foo.dot -o file.png && sxiv file.png
dot -Tpdf cfg.dot -o cfg.pdf
```

# Dot clean
```
digraph "CFG ' function" {
	label="CFG function";
	rankdir=TB;
	splines=ortho;
	concentrate=false;
	ordering=out;
	ranksep=1;
	nodesep=1.0; # Remove if arrow are sticky
```
