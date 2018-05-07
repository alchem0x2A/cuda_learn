all: helloworld

helloworld: main.cu
	nvcc -o $@ $<

clean:
	rm helloworld *.exp *.exe *.lib *.o
