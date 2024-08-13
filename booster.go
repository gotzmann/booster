package main

/*
#cgo linux    CFLAGS: -O3 -g -std=c17   -I.          -fPIC -pthread -march=native -mtune=native -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -DGGML_CUDA_FA_ALL_QUANTS -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin   CFLAGS: -O3 -g -std=c17   -I.          -fPIC -pthread -mcpu=native                -DGGML_USE_LLAMAFILE                   -DGGML_CUDA_FA_ALL_QUANTS -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE
#cgo linux  CXXFLAGS: -O3 -g -std=c++17 -I. -Icommon -fPIC -pthread -march=native -mtune=native -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -DGGML_CUDA_FA_ALL_QUANTS -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin CXXFLAGS: -O3 -g -std=c++17 -I. -Icommon -fPIC -pthread -mcpu=native                -DGGML_USE_LLAMAFILE                   -DGGML_CUDA_FA_ALL_QUANTS -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE
#cgo linux   LDFLAGS: -Lcpp/ggml/src/cuda cpp/bridge.o cpp/janus.o cpp/src/llama.o cpp/common/common.o cpp/common/sampling.o cpp/src/llama-sampling.o cpp/common/build-info.o cpp/common/json-schema-to-grammar.o cpp/common/grammar-parser.o cpp/src/llama-vocab.o cpp/src/llama-grammar.o cpp/src/unicode.o cpp/src/unicode-data.o cpp/ggml/src/ggml.o cpp/ggml/src/ggml-alloc.o cpp/ggml/src/ggml-backend.o cpp/ggml/src/ggml-quants.o cpp/ggml/src/ggml-aarch64.o cpp/ggml/src/llamafile/sgemm.o cpp/ggml/src/ggml-cuda.o -lstdc++ -lm -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -fopenmp -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/lib/wsl/lib
#cgo darwin  LDFLAGS: cpp/bridge.o cpp/janus.o cpp/src/llama.o cpp/common/common.o cpp/common/sampling.o cpp/src/llama-sampling.o cpp/common/build-info.o cpp/common/json-schema-to-grammar.o cpp/common/grammar-parser.o cpp/src/llama-vocab.o cpp/src/llama-grammar.o cpp/src/unicode.o cpp/src/unicode-data.o cpp/ggml/src/ggml.o cpp/ggml/src/ggml-alloc.o cpp/ggml/src/ggml-backend.o cpp/ggml/src/ggml-quants.o cpp/ggml/src/ggml-aarch64.o cpp/ggml/src/llamafile/sgemm.o cpp/ggml/src/ggml-metal.o cpp/ggml/src/ggml-metal-embed.o cpp/ggml/src/ggml-blas.o -lstdc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
*/
import "C"
import "github.com/gotzmann/booster/pkg/booster"

func main() {
	booster.Run()
}
