all: timing

timing: main_test.cu
	nvcc -o $@ $<

clean:
	rm timing *.exp *.exe *.lib *.o
