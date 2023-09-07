# Slow remagnetization of ferrofluids. Effect of chain-like aggregates


all:
	#~/espresso/build/pypresso main.py
	mpirun -n 2 ~/espresso/build/pypresso main.py


# command $make clean
clean:
	rm -rf data/
	rm -rf figs/
	rm -rf vmd/
	mkdir data/
	mkdir figs/
	mkdir vmd/
	cp macos.tcl vmd/macos.tcl
