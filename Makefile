# How to remove older files and build fresh executable?
# make clean && LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j <platform>

# How to run server with debug output?
# ./llamazoo --server --debug

# nvcc --list-gpu-arch
# https://developer.nvidia.com/cuda-gpus
# NVCCFLAGS += -arch=sm_80 -std=c++11

# clean: rm -f *.a bridge.o janus.o llamazoo

# default: llamazoo

clean:
	rm -vrf *.o llama.cpp/*.o *.so *.dll

# How to build for Apple Silicon with CPU + Metal support?
mac: bridge.o janus.o
	cd llama.cpp && \
	make -j macobjs && \
	cd .. && \
	CGO_ENABLED=1 go build llamazoo.go

try: bridge.o janus.o
	cd llama.cpp && \
	echo "== 1 ==" && \
	make clean && \
	echo "== 2 ==" && \
	make -j macobjs && \
	echo "== 3 ==" && \
	cd .. && \
	echo "== 4 ==" && \
	pwd && \
	echo "== 5 ==" && \
	CGO_ENABLED=1 go build llamazoo.go

# How to build for regular Intel / AMD platform with just CPU support
llamazoo: bridge.o janus.o ggml.o ggml-backend.o ggml-alloc.o k_quants.o $(OBJS)
	CGO_ENABLED=1 go build llamazoo.go

# How to build for GPU platrorm with CUDA support?
cuda: bridge.o janus.o ggml.o ggml-backend.o ggml-alloc.o k_quants.o ggml-cuda.o $(OBJS)
	CGO_ENABLED=1 go build llamazoo.go	

# How to build for Apple Silicon with CPU + Metal support?
# mac: bridge.o janus.o ggml.o ggml-backend.o ggml-alloc.o k_quants.o ggml-metal.o $(OBJS)
#	CGO_ENABLED=1 go build llamazoo.go

bridge.o: bridge.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -c $< -o $@

janus.o: janus.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -c $< -o $@
