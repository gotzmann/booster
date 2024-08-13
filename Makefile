# How to remove older files and build fresh executable?
# make clean && LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j <platform>

# How to run server with debug output?
# ./booster --server --debug

# nvcc --list-gpu-arch
# https://developer.nvidia.com/cuda-gpus
# NVCCFLAGS += -arch=sm_80 -std=c++11

# TODO: Detect platform features and choose right default target automatically
# TODO: GGML_OPENBLAS or GGML_OPENBLAS64 ?

default: cuda

# -- Nvidia GPUs with CUDA
cuda:
	cd cpp && \
	GGML_CUDA=on CUDA_USE_GRAPHS=on USE_LLAMAFILE=on make -j cudaobjs && \
	cd .. && \
	CGO_ENABLED=1 go build booster.go

# -- Apple Silicon with both ARM CPU with Neon and GPU Metal support
mac:
	cd cpp && \
	GGML_METAL=on LLAMA_METAL_EMBED_LIBRARY=on USE_LLAMAFILE=on make -j macobjs && \
	cd .. && \
	CGO_ENABLED=1 go build booster.go

# -- Server platforms and Macs with only CPU support
#    TODO: Exclude CUDA drivers as linker requirements
cpu:
	cd cpp && \
	USE_LLAMAFILE=on make -j cpuobjs && \
	cd .. && \
	CGO_ENABLED=1 go build -o booster booster_cpu.go

# -- TODO: OpenCL cards
#    ...

clean:
	rm -vrf *.o *.so *.dll cpp/*.o cpp/src/*.o cpp/ggml/src/*.o cpp/common/*.o cpp/ggml/src/llamafile/*.o
