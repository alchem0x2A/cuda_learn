all: helloworld

helloworld: main.cu my.h
	nvcc -o $@ $<

clean:
	rm helloworld *.exp *.exe *.lib *.o
