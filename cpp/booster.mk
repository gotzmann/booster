cpuobjs: bridge.o janus.o src/llama.o ggml/src/ggml.o ggml-backend.o ggml-alloc.o ggml-quants.o unicode.o unicode-data.o sgemm.o

cudaobjs: bridge.o janus.o \
	common/common.o common/sampling.o src/llama-sampling.o common/build-info.o common/json-schema-to-grammar.o common/grammar-parser.o \
	src/unicode.o src/unicode-data.o src/llama.o src/llama-vocab.o src/llama-grammar.o \
	ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o ggml/src/ggml-blas.o \
	ggml/src/llamafile/sgemm.o \
	ggml/src/ggml-cuda.o \
	$(OBJ_GGML)

macobjs: bridge.o janus.o \
	common/common.o common/sampling.o src/llama-sampling.o common/build-info.o common/json-schema-to-grammar.o common/grammar-parser.o \
	src/unicode.o src/unicode-data.o src/llama.o src/llama-vocab.o src/llama-grammar.o \
	ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o ggml/src/ggml-blas.o \
	ggml/src/llamafile/sgemm.o \
	ggml/src/ggml-metal.o ggml/src/ggml-metal-embed.o

bridge.o: bridge.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -c $< -o $@

janus.o: janus.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -c $< -o $@
