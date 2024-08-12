# How to remove older files and build fresh executable?
# make clean && LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j <platform>

# How to run server with debug output?
# ./booster --server --debug

# nvcc --list-gpu-arch
# https://developer.nvidia.com/cuda-gpus
# NVCCFLAGS += -arch=sm_80 -std=c++11

# -- TODO: Detect platform features and choose right default target automatically

default: cuda

# -- Nvidia GPUs with CUDA
cuda:
	cd cpp && \
	GGML_CUDA=1 CUDA_USE_GRAPHS=1 USE_LLAMAFILE=1 CUDA_FA_ALL_QUANTS=1 make -j cudaobjs && \
	cd .. && \
	CGO_ENABLED=1 go build booster.go

# -- Apple Silicon with both ARM CPU with Neon and GPU Metal support
mac:
	cd cpp && \
	GGML_METAL=1 LLAMA_METAL_EMBED_LIBRARY=1 USE_LLAMAFILE=1 make -j macobjs && \
	cd .. && \
	CGO_ENABLED=1 go build booster.go

# -- Server platforms and Macs with only CPU support
#    TODO: Exclude CUDA drivers as linker requirements
cpu:
	cd cpp && \
	USE_LLAMAFILE=1 make -j cpuobjs && \
	cd .. && \
	CGO_ENABLED=1 go build -o booster booster_cpu.go

# -- TODO: OpenCL cards
#    ...

clean:
	rm -vrf *.o *.so *.dll cpp/*.o cpp/src/*.o cpp/ggml/src/*.o cpp/common/*.o cpp/ggml/src/llamafile/*.o
