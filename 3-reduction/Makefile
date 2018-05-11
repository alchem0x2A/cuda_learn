NAME=reduction
NVCCFLAGS = -arch=sm_21
all: $(NAME)

$(NAME): main.cu
	nvcc $(NVCCFLAGS) -o $@ $<

clean:
	rm $(NAME) *.exp *.exe *.lib *.o
