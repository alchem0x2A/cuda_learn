all: swap

swap: main.cu
	nvcc -o $@ $<

clean:
	rm swap *.exp *.exe *.lib *.o
