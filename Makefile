# How to remove older files and build fresh executable?
# make clean && LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j <platform>

# How to run server with debug output?
# ./collider --server --debug

# nvcc --list-gpu-arch
# https://developer.nvidia.com/cuda-gpus
# NVCCFLAGS += -arch=sm_80 -std=c++11

# -- TODO: Detect platform features and choose right default target automatically

default: cuda clean

# -- Nvidia GPUs with CUDA
cuda:
	cd cpp && \
	LLAMA_CUBLAS=1 make -j cudaobjs && \
	cd .. && \
	CGO_ENABLED=1 go build collider.go

# -- Apple Silicon with both ARM CPU with Neon and GPU Metal support
mac:
	cd cpp && \
	make -j macobjs && \
	cd .. && \
	CGO_ENABLED=1 go build collider.go

# -- Server platforms and Macs with only CPU support
#    TODO: Exclude CUDA drivers as linker requirements
cpu:
	cd cpp && \
	LLAMA_NO_METAL=1 make -j cpuobjs && \
	cd .. && \
	CGO_ENABLED=1 go build -o collider collider_cpu.go

# -- Nvidia GPUs with CUDA
cuda:
	cd cpp && \
	LLAMA_CUBLAS=1 make -j cudaobjs && \
	cd .. && \
	CGO_ENABLED=1 go build collider.go

# -- TODO: OpenCL cards
#    ...

clean:
	rm -vrf *.o cpp/*.o *.so *.dll
